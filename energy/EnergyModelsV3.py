import torch
from argparse import Namespace
from torch import nn

from energy.AbstractEnergyModels import AbstractTimeseriesEBMV2
from energy.EnergyModelsV2 import ArbitraryMLPDecoder, MultisampleMLPDecoder
from layers.AutoCorrelation import AutoCorrelationLayer, AutoCorrelation
from layers.Autoformer_EncDec import DecoderLayer, Decoder, my_Layernorm
from models import Autoformer, Informer


class AutoformerNeoEBM(AbstractTimeseriesEBMV2):
    def __init__(
        self,
        setting: Namespace,
        autoformer: Autoformer,
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
        
        
        super(AutoformerNeoEBM, self).__init__(uses_flat_xs=False)

        self.setting = setting
        self.autoformer: Autoformer.Model = autoformer

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.lstm_hidden_size = x_encoder_lstm_hidden_size
        self.predictor_code_dim = x_encoder_code_dim
        self.lstm_num_layers = x_encoder_lstm_num_layers

        self.x_decoder_num_layers = x_decoder_num_layers
        self.x_decoder_code_dim = x_decoder_code_dim

        self.y_decoder_num_layers = y_decoder_num_layers
        self.y_decoder_code_dim = y_decoder_code_dim

        self.decoder_hidden_dim = xy_decoder_hidden_dim
        self.decoder_num_layers = xy_decoder_num_layers

        self.batch_samples = batch_samples

        
        self.c = nn.Parameter(
            torch.tensor([1.0], requires_grad=use_normalizing_constant)
        )

        self.y_encoder = None
        self.xy_decoder = None

    
    
    def setup_y_encoder_and_xy_decoder_(
        self, seq_len, label_len, pred_len, d_model, c_out, dec_out_2_dim
    ):
        

        self.y_encoder = MultisampleMLPDecoder(
            
            input_dim=(dec_out_2_dim) * c_out,
            output_dim=seq_len * d_model,
            num_layers=self.y_decoder_num_layers,
            hidden_size=self.y_decoder_code_dim,
        )
        self.xy_decoder = ArbitraryMLPDecoder(
            input_dim=seq_len * d_model,
            output_dim=1,  
            num_layers=self.decoder_num_layers,
            hidden_size=self.decoder_hidden_dim,
        )
        self.orig_model_seq_len = seq_len
        self.orig_model_label_len = label_len
        self.orig_model_pred_len = pred_len
        self.orig_model_d_model = d_model

        print(f"Setup for of Y encoder and XY decoder done!")

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

    def _forward_enc_orig(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        return self.autoformer.forward_enc_(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            enc_self_mask,
            dec_self_mask,
            dec_enc_mask,
        )

    def _forward_dec_orig(
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
        return self.autoformer.forward_dec_(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            enc_out,
            enc_self_mask,
            dec_self_mask,
            dec_enc_mask,
        )

    def _forward_y_enc(self, batch_y):
        actual_y = batch_y[:, -self.orig_model_pred_len :, :]
        
        reshaped_batch_y = torch.cat(
            [torch.zeros_like(batch_y[:, : -self.orig_model_pred_len, :]), actual_y],
            dim=1,
        )
        encoded_y = self.y_encoder(reshaped_batch_y)
        reshaped_encoded_y = encoded_y.reshape(
            reshaped_batch_y.shape[0], self.orig_model_seq_len, self.orig_model_d_model
        )
        
        return reshaped_encoded_y

    def get_decoded(self, xs: torch.Tensor, ys: torch.Tensor):
        encoded_x = self.get_encoded_x(xs)  
        return self._get_decoded(encoded_x, ys)

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

        

        
        
        
        
        
        
        
        
        

        
        
        xy_encoded = torch.sub(encoded_x, encoded_y)
        xy_encoded_reshaped = xy_encoded.view(xy_encoded.shape[0], -1)
        score = self.xy_decoder(xy_encoded_reshaped)

        
        return score

    def forward(self, xs: torch.Tensor, ys: torch.Tensor):
        return self.get_decoded(xs, ys)


class AutoformerNeoEBM_concat(AutoformerNeoEBM):
    def __init__(
        self,
        setting: Namespace,
        autoformer: Autoformer,
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
        super(AutoformerNeoEBM_concat, self).__init__(
            setting,
            autoformer,
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

    
    
    def setup_y_encoder_and_xy_decoder_(
        self, seq_len, label_len, pred_len, d_model, c_out, dec_out_2_dim
    ):
        super().setup_y_encoder_and_xy_decoder_(
            seq_len, label_len, pred_len, d_model, c_out, dec_out_2_dim
        )
        self.xy_decoder = ArbitraryMLPDecoder(
            input_dim=seq_len * d_model * 2,  
            output_dim=1,  
            num_layers=self.decoder_num_layers,
            hidden_size=self.decoder_hidden_dim,
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
        xy_encoded = torch.cat([encoded_x, encoded_y], dim=2)
        xy_encoded_reshaped = xy_encoded.view(xy_encoded.shape[0], -1)
        score = self.xy_decoder(xy_encoded_reshaped)

        
        return score


class AutoformerNeoEBM_transformer(AutoformerNeoEBM):
    def __init__(
        self,
        setting: Namespace,
        autoformer: Autoformer,
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
        super(AutoformerNeoEBM_transformer, self).__init__(
            setting,
            autoformer,
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

        
        self.xy_decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                            use_gpu=configs.use_gpu,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                            use_gpu=configs.use_gpu,
                        ),
                        configs.d_model,
                        configs.n_heads,
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
            torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.autoformer.pred_len, 1)
        )
        zeros = torch.zeros(
            [x_dec.shape[0], self.autoformer.pred_len, x_dec.shape[2]],
            device=x_enc.device,
        )
        seasonal_init, trend_init = self.autoformer.decomp(x_enc)
        
        trend_init = torch.cat(
            [trend_init[:, -self.autoformer.label_len :, :], mean], dim=1
        )
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.autoformer.label_len :, :], zeros], dim=1
        )

        
        dec_out = self.autoformer.dec_embedding(seasonal_init, x_mark_dec)
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

    
    
    
    
    
    
    
    






class InformerNeoEBM(AutoformerNeoEBM):
    def __init__(
        self,
        setting: Namespace,
        informer: Informer,
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
        super(InformerNeoEBM, self).__init__(
            setting,
            informer,
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
        self.informer = informer

    def get_assumed_enc_out_shape(self, args):
        assumed_enc_out_shape = [
            args.batch_size,
            args.seq_len // 2 + args.seq_len % 2 + 1,
            args.d_model,
        ]
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
        

        self.y_encoder_output_dim = seq_len // 2 + 1
        self.y_encoder = MultisampleMLPDecoder(
            
            input_dim=(dec_out_2_dim) * c_out,
            output_dim=self.y_encoder_output_dim
            * d_model,  
            num_layers=self.y_decoder_num_layers,
            hidden_size=self.y_decoder_code_dim,
        )
        self.xy_decoder = ArbitraryMLPDecoder(
            input_dim=self.y_encoder_output_dim * 2,
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
            (self.y_encoder_output_dim),
            self.orig_model_d_model,
        )
        return reshaped_encoded_y


class InformerNeoEBM_concat(InformerNeoEBM):
    def __init__(
        self,
        setting: Namespace,
        informer: Informer,
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
        super(InformerNeoEBM_concat, self).__init__(
            setting,
            informer,
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

    
    
    def setup_y_encoder_and_xy_decoder_(
        self, seq_len, label_len, pred_len, d_model, c_out, dec_out_2_dim
    ):
        super().setup_y_encoder_and_xy_decoder_(
            seq_len, label_len, pred_len, d_model, c_out, dec_out_2_dim
        )
        self.xy_decoder = ArbitraryMLPDecoder(
            input_dim=(seq_len // 2 + seq_len % 2 + 1)
            * d_model
            * 2,  
            output_dim=1,  
            num_layers=self.decoder_num_layers,
            hidden_size=self.decoder_hidden_dim,
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
        xy_encoded = torch.cat([encoded_x, encoded_y], dim=2)
        xy_encoded_reshaped = xy_encoded.view(xy_encoded.shape[0], -1)
        score = self.xy_decoder(xy_encoded_reshaped)

        
        return score


class InformerNeoEBM_transformer(InformerNeoEBM):
    def __init__(
        self,
        setting: Namespace,
        informer: Informer,
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
        super(InformerNeoEBM_transformer, self).__init__(
            setting,
            informer,
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

        
        from layers.Transformer_EncDec import (
            Decoder,
            DecoderLayer,
        )
        from layers.SelfAttention_Family import (
            ProbAttention,
            AttentionLayer,
        )

        self.xy_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        ProbAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
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

        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        

        dec_out = self.informer.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.xy_decoder(
            dec_out, encoded_xy, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )

        dec_out = dec_out.view(dec_out.shape[0], -1)
        score = self.decoder_energy_linear(dec_out)

        return score
