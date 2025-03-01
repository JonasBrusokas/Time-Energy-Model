import torch
import torch.nn as nn

# %%
# https://discuss.pytorch.org/t/rmse-loss-function/16540/3
class root_mean_squared_error(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

class maximum_absolute_error(nn.Module):
    def forward(self, yhat, y):
        return torch.max(torch.abs(torch.sub(y, yhat)))

