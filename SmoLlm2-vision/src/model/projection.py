from typing import Literal

import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int, projection_type: Literal['mlp', 'linear'] = 'mlp'):
        super().__init__()
        
        if projection_type == 'linear':
            self.projection = nn.Linear(input_dim, output_dim)
        elif projection_type == 'mlp':
            self.projection = nn.Sequential(
                nn.Linear(input_dim, output_dim * 4),
                nn.GELU(),
                nn.Linear(output_dim * 4, output_dim)
            )
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")
        
    
    def forward(self, x):
        return self.projection(x)
