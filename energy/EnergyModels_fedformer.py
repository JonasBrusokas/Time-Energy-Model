import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

from energy.EnergyModelsV2 import ArbitraryMLPDecoder, MultisampleMLPDecoder
from energy.EnergyModelsV3 import AutoformerNeoEBM
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FEDformer.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.FEDformer.MultiWaveletCorrelation import (
    MultiWaveletCross,
    MultiWaveletTransform,
)
from models import FEDformer


class FedformerNeoEBM_concat(AutoformerNeoEBM):
    def __init__(
        self,
        setting: Namespace,
        fedformer: FEDformer,
        x_dim: int,  
        y_dim: int,
        x_encoder_lstm_hidden_size: int = 16,
        x_encoder_code_dim: int = 8,
        x_encoder_lstm_num_layers: int = 1,
        x_decoder_num_layers: int = 1,
        x_decoder_code_dim: int = 8,
        y_decoder_num_layers: int = 1,
        y_decoder_code_dim: int = 8,
        xy_decoder_hidden_dim: int = 128,
        xy_decoder_num_layers: int = 3,
        use_normalizing_constant: bool = False,
        batch_samples: bool = False,
    ):
        super(FedformerNeoEBM_concat, self).__init__(
            setting,
            fedformer,
            x_dim,
            y_dim,
            x_encoder_lstm_hidden_size,
            x_encoder_code_dim,
            x_encoder_lstm_num_layers,
            x_decoder_num_layers,
            x_decoder_code_dim,
            y_decoder_num_layers,
            y_decoder_code_dim,
            xy_decoder_hidden_dim,
            xy_decoder_num_layers,
            use_normalizing_constant,
            batch_samples,
        )
        self.fedformer = fedformer

    def get_assumed_enc_out_shape(self, args):
        assumed_enc_out_shape = [args.batch_size, args.seq_len, args.d_model]
        return assumed_enc_out_shape

    def get_assumed_dec_out_shape(self, args):
        assumed_dec_out_shape = [
            args.batch_size,
            args.label_len + args.pred_len,
            args.c_out,
        ]
        return assumed_dec_out_shape

    
    
    def setup_y_encoder_and_xy_decoder_(
        self, seq_len, label_len, pred_len, d_model, c_out, dec_out_2_dim
    ):
        

        self.y_encoder = MultisampleMLPDecoder(
            
            input_dim=(dec_out_2_dim) * c_out,
            output_dim=(seq_len) * d_model,
            num_layers=self.y_decoder_num_layers,
            hidden_size=self.y_decoder_code_dim,
        )
        self.xy_decoder = ArbitraryMLPDecoder(
            input_dim=(seq_len * 2)
            * d_model,  
            output_dim=1,  
            num_layers=self.decoder_num_layers,
            hidden_size=self.decoder_hidden_dim,
        )
        self.orig_model_seq_len = seq_len
        self.orig_model_label_len = label_len
        self.orig_model_pred_len = pred_len
        self.orig_model_d_model = d_model

        print(f"Setup for of Y encoder and XY decoder done!")

    def _forward_y_enc(self, batch_y):
        actual_y = batch_y[:, -self.orig_model_pred_len :, :]
        reshaped_batch_y = torch.cat(
            [torch.zeros_like(batch_y[:, : -self.orig_model_pred_len, :]), actual_y],
            dim=1,
        )
        encoded_y = self.y_encoder(reshaped_batch_y)
        reshaped_encoded_y = encoded_y.reshape(
            reshaped_batch_y.shape[0],
            (self.orig_model_seq_len),
            self.orig_model_d_model,
        )
        return reshaped_encoded_y

    def _get_decoded(
        self,
        batch_y: torch.Tensor,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_out,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        encoded_x = enc_out
        encoded_y = self._forward_y_enc(
            batch_y
        )  
        xy_encoded = torch.cat([encoded_x, encoded_y], dim=2)
        xy_encoded_reshaped = xy_encoded.view(xy_encoded.shape[0], -1)
        score = self.xy_decoder(xy_encoded_reshaped)

        
        return score


class FedformerNeoEBM_transformer(FedformerNeoEBM_concat):
    def __init__(
        self,
        setting: Namespace,
        fedformer: FEDformer,
        x_dim: int,  
        y_dim: int,
        configs,
        x_encoder_lstm_hidden_size: int = 16,
        x_encoder_code_dim: int = 8,
        x_encoder_lstm_num_layers: int = 1,
        x_decoder_num_layers: int = 1,
        x_decoder_code_dim: int = 8,
        y_decoder_num_layers: int = 1,
        y_decoder_code_dim: int = 8,
        xy_decoder_hidden_dim: int = 128,
        xy_decoder_num_layers: int = 3,
        use_normalizing_constant: bool = False,
        batch_samples: bool = False,
    ):
        super(FedformerNeoEBM_transformer, self).__init__(
            setting,
            fedformer,
            x_dim,
            y_dim,
            x_encoder_lstm_hidden_size,
            x_encoder_code_dim,
            x_encoder_lstm_num_layers,
            x_decoder_num_layers,
            x_decoder_code_dim,
            y_decoder_num_layers,
            y_decoder_code_dim,
            xy_decoder_hidden_dim,
            xy_decoder_num_layers,
            use_normalizing_constant,
            batch_samples,
        )
        self.configs = configs

    
    
    def setup_y_encoder_and_xy_decoder_(
        self, seq_len, label_len, pred_len, d_model, c_out, dec_out_2_dim
    ):
        super().setup_y_encoder_and_xy_decoder_(
            seq_len, label_len, pred_len, d_model, c_out, dec_out_2_dim
        )

        configs = self.configs

        

        
        if configs.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base
            )
            decoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base
            )
            decoder_cross_att = MultiWaveletCross(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.fedformer.seq_len // 2 + self.fedformer.pred_len,
                seq_len_kv=self.fedformer.seq_len,
                modes=configs.modes,
                ich=configs.d_model,
                base=configs.base,
                activation=configs.cross_activation,
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.fedformer.seq_len,
                modes=configs.modes,
                mode_select_method=configs.mode_select,
            )
            decoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.fedformer.seq_len // 2 + self.fedformer.pred_len,
                modes=configs.modes,
                mode_select_method=configs.mode_select,
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.fedformer.seq_len // 2 + self.fedformer.pred_len,
                seq_len_kv=self.fedformer.seq_len,
                modes=configs.modes,
                mode_select_method=configs.mode_select,
            )

        from layers.FEDformer.Autoformer_EncDec import (
            Decoder,
            DecoderLayer,
            my_Layernorm,
        )

        self.xy_decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att, configs.d_model, configs.n_heads
                    ),
                    AutoCorrelationLayer(
                        decoder_cross_att, configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

        self.decoder_energy_linear = nn.Linear(
            in_features=dec_out_2_dim, out_features=1
        )

        print(f"Setup for of Y encoder and XY decoder done!")

    def _get_decoded(
        self,
        batch_y: torch.Tensor,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_out,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        encoded_x = enc_out
        encoded_y = self._forward_y_enc(
            batch_y
        )  
        encoded_xy = torch.sub(encoded_x, encoded_y)

        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        

        
        mean = (
            torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.fedformer.pred_len, 1)
        )
        seasonal_init, trend_init = self.fedformer.decomp(x_enc)
        
        trend_init = torch.cat(
            [trend_init[:, -self.fedformer.label_len :, :], mean], dim=1
        )
        seasonal_init = F.pad(
            seasonal_init[:, -self.fedformer.label_len :, :],
            (0, 0, 0, self.fedformer.pred_len),
        )
        
        dec_out = self.fedformer.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.xy_decoder(
            dec_out,
            encoded_xy,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            trend=trend_init,
        )
        
        dec_out = trend_part + seasonal_part
        dec_out = dec_out.view(dec_out.shape[0], -1)
        score = self.decoder_energy_linear(dec_out)

        return score
