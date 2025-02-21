import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler

from loguru import logger
from tqdm import tqdm

from metrics import get_criterion


class WarmUpLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, initial_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]

def pre_training(model, dataset, config, checkpoint_dir, device):
    """
    pre train model on unlabeled masked data
    """
    model.train()

    #---load params from config
    lr = config["lr"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    criterion_name = config["criterion"]
    criterion, supported_criterion = get_criterion(criterion_name)
    if not supported_criterion:
        logger.error(f"Criterion {criterion_name} not supported. Using MSEloss instead.")

    optimizer_name = config["optimizer"]
    optimizer, supported_optimizer = get_optimizer(optimizer_name, model.parameters(), lr)
    if not supported_optimizer:
        logger.error(f"Optimizer {optimizer_name} not supported. Using Adam instead.")


    scheduler =  WarmUpLR(optimizer, 5, lr)
    
    #data loader
    logger.info(f"Creating dataloaders with batch size {batch_size}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

    logger.info(f"Starting pre-training for {num_epochs} epochs")
    best_loss = float("inf")
    losses = []
    lrates= []
    for epoch in range(num_epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            # if epoch == 10:
            #     #change mask proportion after 10 epochs
            #     model.patch_encoding.mask_proportion = 0.75

            img, patches = batch
            img = img.to(device)
            patches = patches.to(device)
        
            optimizer.zero_grad()

            reconstructed_images = model(patches)
          
            loss = criterion(reconstructed_images, patches)
            loss.backward()

            utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Monitor gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            optimizer.step()
            
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Gradient Norm: {total_norm}")
        losses.append(loss.item())

        # Step the scheduler
        scheduler.step()
        lrates.append(optimizer.param_groups[0]['lr'])

        #save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{checkpoint_dir}/last_epoch.pt")
        if loss < best_loss:
            best_loss = loss
            torch.save(checkpoint, f"{checkpoint_dir}/best_loss.pt")

    return model, losses, lrates

def fine_tuning(model, dataset, config, device):
    """
    fine tune model (freeze encoder) on labeled data
    """
    model.train()

    #---load params from config
    lr = config["lr"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    criterion_name = config["criterion"]
    criterion, supported_criterion = get_criterion(criterion_name)
    if not supported_criterion:
        logger.error(f"Criterion {criterion_name} not supported. Using MSEloss instead.")

    optimizer_name = config["optimizer"]
    optimizer, supported_optimizer = get_optimizer(optimizer_name, model.parameters(), lr)
    if not supported_optimizer:
        logger.error(f"Optimizer {optimizer_name} not supported. Using Adam instead.")
    
    #split the dataset into train and val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    #data loader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            img, labels = batch 
            img = img.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(img)
            loss = criterion(output, labels)    
            loss.backward()

            optimizer.step()
            
        logger.info(f"Epoch {epoch}/{num_epochs}, Train Loss: {loss.item()}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                img, labels = batch
                img = img.to(device)
                labels = labels.to(device)

                output = model(img)
                loss = criterion(output, labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        logger.info(f"Epoch {epoch}/{num_epochs}, Validation Loss: {val_loss}")

    return model


def get_optimizer(name, params, lr):
    """
    pick the optimizer to use for the training.
    args:
        name: str, the name of the optimizer to use.    
    """
    name = name.lower()
    supported_optimizer = True
    if "adam" in name:
        optimizer = torch.optim.Adam(params, lr)
    elif "sgd" in name:
        optimizer = torch.optim.SGD(params, lr)
    else:
        optimizer = torch.optim.Adam(params, lr)
        supported_optimizer = False

    return optimizer, supported_optimizer