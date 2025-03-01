from torch import nn

from utilz import *


class AbstractModel(nn.Module):

    def __init__(self):
        super(AbstractModel, self).__init__()

    def forward_enc_(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                     enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        throw_noimpl()