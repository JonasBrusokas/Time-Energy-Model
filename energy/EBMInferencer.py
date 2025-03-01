

import torch
import torch.nn as nn




class EBMInferencer(nn.Module):
    
    def __init__(self, ebm, initial_tensor: torch.tensor):
        super().__init__()
        self.ebm = ebm
        self.y_hat = nn.Parameter(
            initial_tensor
        )

    def forward(self, X):
        energy = self.ebm(X, self.y_hat)
        return energy