import torch
from torch import nn

from utilz import *


class AbstractTimeseriesEBM(nn.Module):

    def __init__(self,
                 uses_flat_xs: bool,
                 ):
        super(AbstractTimeseriesEBM, self).__init__()
        self.uses_flat_xs = uses_flat_xs

    def need_flatten_xs(self) -> bool:
        return self.uses_flat_xs

    def get_encoded_x(self, xs: torch.Tensor) -> torch.Tensor:
        throw_noimpl()

    def get_decoded(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        throw_noimpl()

class AbstractTimeSeriesEBMDecoderMixin:

    def get_final_representation(self, x_encoded, y):
        throw_noimpl()

class AbstractTimeSeriesEBMPredictorMixin:
    pass
    
    
    

class AbstractTimeSeriesEBMPlugin(nn.Module):

    def __init__(self, ebm: AbstractTimeseriesEBM):
        super(AbstractTimeSeriesEBMPlugin, self).__init__()
        self.ebm = ebm

    def need_flatten_xs(self) -> bool:
        return self.ebm.uses_flat_xs

class AbstractTimeseriesEBMV2(AbstractTimeseriesEBM):

    
    
    

    """
    This class describes the alternate architecture of EBMs where we use Supervised Learning
    to train X and Y encoders and then Self-supervised Learning to train the XY decoder.
    """

    def __init__(self,
                 uses_flat_xs: bool,
                 ):
        super(AbstractTimeseriesEBMV2, self).__init__(uses_flat_xs=uses_flat_xs)

    def get_decoded_x(self, encoded_x: torch.Tensor) -> torch.Tensor:
        throw_noimpl()

    def get_encoded_y(self, y: torch.Tensor) -> torch.Tensor:
        throw_noimpl()

    def get_decoded_y(self, encoded_y: torch.Tensor) -> torch.Tensor:
        throw_noimpl()

    def get_decoded_xy(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        return self.get_decoded(xs, ys)
