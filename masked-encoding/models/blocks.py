import torch
import torch.nn as nn

from einops import rearrange

class PatchEncoding(nn.Module):
    def __init__(self, patch_size, num_patches, mask_proportion, embedding_dim, in_channels=3):
        super(PatchEncoding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mask_proportion = mask_proportion
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        self.projection = nn.Linear(in_channels * patch_size * patch_size, embedding_dim)

        #learnable positional embeddings
        self.position_embeddings = nn.Parameter(torch.empty(1, num_patches*num_patches, embedding_dim))
        nn.init.xavier_uniform_(self.position_embeddings)

        #learnable mask token, for the decoder part
        self.mask_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, x):
        device = x.device

        #x shape: (batch_size, num_patches, in_channel, height, width)
        batch_size, n_patches, in_ch, height, width = x.shape

        position_embeddings = self.position_embeddings.expand(batch_size, -1, -1).to(device)

        assert in_ch == self.in_channels, f"Input channels should be {self.in_channels}, got {in_ch}"
        assert n_patches == self.num_patches*self.num_patches, f"Number of patches should be {self.num_patches*self.num_patches}, got {n_patches}"
  
        patches = rearrange(x, 'b n c h w -> b n (c h w)')

        patch_embeddings = self.projection(patches)
        patch_embeddings = patch_embeddings + position_embeddings

        #sample masked patches
        num_masked = int(n_patches * self.mask_proportion)  
        rand_indices = torch.rand(batch_size, n_patches, device=device).argsort(dim=-1)
        mask_indices = rand_indices[:, :num_masked]
        unmask_indices = rand_indices[:, num_masked:]

        mask_indices = mask_indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
        unmask_indices = unmask_indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)

        #separate embeddings that will go through the encoder from the ones that will be masked
        unmasked_embeddings = torch.gather(patch_embeddings, 1, unmask_indices)
        masked_embeddings = self.mask_token.expand(batch_size, num_masked, -1) 

        return unmasked_embeddings, masked_embeddings, position_embeddings, mask_indices, unmask_indices