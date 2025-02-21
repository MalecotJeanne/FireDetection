import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, patch_size, embedding_dim, config):
        super(TransformerDecoder, self).__init__()

        self.embedding_dim = embedding_dim
        num_heads = config["num_heads"]
        num_layers = config["num_layers"]
        mlp_ratio = config["mlp_ratio"]
        dropout = config["dropout"]
        self.patch_size = patch_size

        self.decoder_layers = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=mlp_ratio * embedding_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True  
            ),
            num_layers=num_layers
        )
        
        self.norm = nn.LayerNorm(embedding_dim)
        # projection to match patch size
        self.output_projection = nn.Linear(embedding_dim, patch_size * patch_size * 3)  

    def forward(self, encoded_input, memory):
        """
        encoded_input: (batch_size, num_patches, embedding_dim) - input to the decoder
        memory: (batch_size, num_patches, embedding_dim) - output of the encoder
        """
        output = self.decoder_layers(encoded_input, memory)
        output = self.norm(output)  
        output = self.output_projection(output)

        output = output.view(output.shape[0], output.shape[1], 3, self.patch_size, self.patch_size)
        
        output = F.tanh(output)  # tanh activation function to match the range of pixel values
        return output
