import os
import sys

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

import torch
import torch.nn as nn
import torch.nn.init as init

from models.blocks import PatchEncoding

class MAE(nn.Module):
    def __init__(self, encoder, decoder, config):
        super(MAE, self).__init__()

        image_dim = config["image_dim"]
        patch_size = image_dim // config["n_patches"]

        self.patch_encoding = PatchEncoding(patch_size, config["n_patches"], config["mask_proportion"], config["embedding_dim"])

        self.encoder = encoder(config["embedding_dim"], config["encoder"])
        self.decoder = decoder(patch_size, config["embedding_dim"], config["decoder"])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0)


    def _get_masked_image(self, x, mask_indices):

        all_patches = x.clone()
        masked_patch = torch.ones(x.shape[-3:]).to(x.device)
        masked_patch = masked_patch * torch.min(x)

        mask_indices = mask_indices[:, :, 0]

        for i in range(x.shape[0]):
            for j in mask_indices[i]:
                all_patches[i, j] = masked_patch

        return all_patches
      
    def forward(self, x, return_masked=False):

        unmasked_embeddings, masked_embeddings, position_embeddings, mask_indices, unmask_indices = self.patch_encoding(x)
        
        num_patches = self.patch_encoding.num_patches * self.patch_encoding.num_patches
        embedding_dim = self.patch_encoding.embedding_dim
        batch_size = x.shape[0]

        #encoder
        encoded = self.encoder(unmasked_embeddings)
        pos_encoded = encoded +  torch.gather(position_embeddings, 1, unmask_indices)
        masked_embeddings = masked_embeddings + torch.gather(position_embeddings, 1, mask_indices)

        #decoder
        all_patches = torch.zeros(batch_size, num_patches, embedding_dim).to(x.device)
        
        all_patches.scatter_(1, unmask_indices, pos_encoded)
        all_patches.scatter_(1, mask_indices, masked_embeddings)

        decoded = self.decoder(all_patches, encoded)

        if return_masked:
            masked = self._get_masked_image(x, mask_indices)
            return decoded, masked

        return decoded