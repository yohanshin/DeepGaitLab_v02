import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.nn import (TransformerDecoder, 
                      TransformerEncoder, 
                      TransformerEncoderLayer, 
                      TransformerDecoderLayer)

class MultiViewEncoder(nn.Module):
    def __init__(self,
                 d_model=768,
                 n_heads=8,
                 n_layers=2,
                 transformer_dim_feedforward=1024,
                 dropout=0.2,
                 width=24,
                 height=32,
                 max_num_cameras=10):
        super(MultiViewEncoder, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.dropout = dropout
        self.max_num_cameras = max_num_cameras

        self.query_embed = nn.Embedding(max_num_cameras, d_model)
        self.bbox_embed = nn.Linear(3, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, height*width + 1, d_model))
        decoder_layer = TransformerDecoderLayer(d_model=d_model, 
                                                nhead=n_heads, 
                                                dim_feedforward=transformer_dim_feedforward,
                                                dropout=dropout, 
                                                batch_first=False)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, 
                                          num_layers=n_layers, 
                                          norm=decoder_norm)

        encoder_layer = TransformerEncoderLayer(d_model=d_model, 
                                                nhead=n_heads, 
                                                dim_feedforward=transformer_dim_feedforward,
                                                dropout=dropout, 
                                                batch_first=False)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, 
                                          num_layers=n_layers, 
                                          norm=encoder_norm)
        
    def forward(self, x, bbox, valid=None):
        # x: B, V, D, H, W
        # bbox: B, V, 3
        # valid: B, V
        
        B, V = bbox.shape[:2]
        # Reshape to (H*W, B*V, D) for transformer
        x = rearrange(x, "(B V) D H W -> (H W) (B V) D", B=B, V=V)
        
        # Get camera queries for each view
        camera_queries = self.query_embed.weight[:V]  # V, D
        camera_queries = camera_queries.unsqueeze(0).repeat(B, 1, 1)  # B, V, D
        
        # Process bbox embeddings
        bbox_embeddings = self.bbox_embed(bbox)  # B, V, D
        
        # Combine camera queries and bbox embeddings
        view_features = camera_queries + bbox_embeddings  # B, V, D
        view_features = rearrange(view_features, "B V D -> (B V) D").unsqueeze(0)
        
        # Add positional embedding
        pos_embed = self.pos_embed[:, 1:]  # 1, H*W, D
        x = x + pos_embed.permute(1, 0, 2)  # H*W, 1, D

        # Decode the global feature
        global_x = self.decoder(tgt=view_features, memory=x)
        global_x = rearrange(global_x.squeeze(0), "(B V) D -> V B D", B=B, V=V)
        
        # Apply transformer encoder
        if valid is not None:
            valid = valid.bool().squeeze(-1)
            # Create padding mask for transformer
            # True values in mask indicate positions to mask out
            padding_mask = ~valid  # B, V
        else:
            padding_mask = None
            
        # Apply transformer
        encoded_features = self.encoder(global_x, src_key_padding_mask=padding_mask)  # H*W, B*V, D
        
        # Reshape back to original format
        encoded_features = rearrange(encoded_features, "V B D -> B V D")
        encoded_features = encoded_features + self.pos_embed[:, :1]
        encoded_features = rearrange(encoded_features, "B V (null D) -> null (B V) D", null=1)
        
        return encoded_features
        