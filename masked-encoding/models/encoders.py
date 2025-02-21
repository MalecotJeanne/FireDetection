import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, config):
        super(TransformerEncoder, self).__init__()

        num_heads = config["num_heads"]
        num_layers = config["num_layers"]
        mlp_ratio = config["mlp_ratio"]
        dropout = config["dropout"]

        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
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

    def forward(self, x):
        x = self.encoder_layers(x)
        return self.norm(x)
    

class CNNEncoder(nn.Module):
    def __init__(self, embedding_dim, config):
        super(CNNEncoder, self).__init__()

        num_layers = config["num_layers"]
        conv_channels = config["conv_channels"]
        kernel_size = config["kernel_size"]
        dropout = config["dropout"]

        layers = []
        in_channels = embedding_dim
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(in_channels, conv_channels, kernel_size=kernel_size, padding=1)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = conv_channels

        #keep embedding_dim size
        layers.append(nn.Conv1d(conv_channels, embedding_dim, kernel_size=1))
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        #x shape: (batch_size, num_patches, embedding_dim)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        return x
    
