import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding
from layers.TimeSeriesLibrary.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    
    xf = torch.fft.rfft(x, dim=1)
    
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        
        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model, configs.d_ff, num_kernels=configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff, configs.d_model, num_kernels=configs.num_kernels
            ),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            
            out = self.conv(out)
            
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = "long_term_forecast"  
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList(
            [TimesBlock(configs) for _ in range(configs.e_layers)]
        )
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        
        
        
        
        
        
        
        
        else:
            raise ValueError(f"Task '{self.task_name}' unsupported!")
            return None

    def forward_enc_(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1
        )  
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        attns = None  
        return enc_out, attns

    def forward_dec_(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_out,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        
        dec_out = self.projection(enc_out)

        
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1
        )  
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        dec_out = self.projection(enc_out)

        
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out, attns = self.forward_enc_(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
        )
        dec_out = self.forward_dec_(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            enc_out,
        )
        return dec_out[:, -self.pred_len :, :]
