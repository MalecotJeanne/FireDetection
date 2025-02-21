"""
executable script for pretraining the masked autoencoder
"""
import argparse
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import yaml
from loguru import logger

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from training import pre_training
from models import init_model
from transforms import training_transforms
from dataset import FireDataset, Patches
from utils import load_config

# Arguments parsing

parser = argparse.ArgumentParser(description='Pretrain a masked autoencoder')
parser.add_argument('-d', '--data', type=str, default='data', help='path to the data folder')
parser.add_argument('-m', '--model', type=str, choices = ["mae", "vit_base", "vit_small"], help='name of the model to train')
parser.add_argument('-c', '--config', type=str, default='config.yaml', help='path to the configuration file')
parser.add_argument('-o', '--output', type=str, default='Pretrained_models', help='path to the output folder')

args = parser.parse_args()


def main():
    
    #output folder
    output_folder = os.path.join(args.output, f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_folder, exist_ok=True)
    
    #logger
    logger_path = os.path.join(output_folder, 'pretraining.log')

    with open(logger_path, "w") as log_file:
        log_file.write("==========================================\n")
        log_file.write("Logs for masked autoencoder pre-training\n")
        log_file.write(f"--- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        log_file.write("----------\n")
        log_file.write(f"Model: {args.model}\n")
        log_file.write("==========================================\n")
        log_file.write("\n")

    logger.add(logger_path, mode = 'a', rotation="10 MB")

    #device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    #load config
    with open(args.config, 'r') as stream:
        config = load_config(yaml.safe_load(stream))

    logger.info(f"Config: {args.config}")
        
    #init the model
    model = init_model('mae', 'transformer', 'transformer', config = config).to(device)

    #create dataset
    transforms = training_transforms(config["transforms"])
    im_dataset = FireDataset(args.data, dataset = "train", transform = transforms)
    dataset = Patches(im_dataset, config["pre-training"]["n_patches"])

    #checkpoint dir
    chkpt_dir = os.path.join(output_folder, "checkpoints")
    os.makedirs(chkpt_dir, exist_ok=True)
    
    # Pretrain the model
    pt_model, losses, lrates = pre_training(model, dataset, config["pre-training"], chkpt_dir, device)

    #losses evolution plot
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss evolution during pre-training")
    plt.savefig(os.path.join(output_folder, "losses.png"))
    plt.close()

    #lr evolution plot
    plt.figure()
    plt.plot(lrates)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning rate evolution during pre-training")
    plt.savefig(os.path.join(output_folder, "lrates.png"))
    plt.close()

    # Save the model
    torch.save(pt_model.state_dict(), os.path.join(output_folder, f"{args.model}_pretrained.pt"))
    
    logger.success(f"Pretraining done! Check the output folder {output_folder} for the results.")


if __name__ == "__main__":
    main()
