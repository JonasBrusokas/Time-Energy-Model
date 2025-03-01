from energy.AbstractEnergyModels import *
from energy.EnergyModels import DecoderEBM, PredictorEBM, ResNetSequential

class ArbitraryMLPDecoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_layers: int,
                 hidden_size: int,
                 activation_foo = nn.LeakyReLU(0.1),
                 ):

        super(ArbitraryMLPDecoder, self).__init__()
        self.activation_foo = activation_foo
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.input_layers = [
            nn.Linear(in_features=self.input_dim,
                      out_features=self.hidden_size),
            self.activation_foo,
        ]

        self.hidden_layers = [ResNetSequential([
            nn.Linear(in_features=self.hidden_size,
                      out_features=self.hidden_size),
            activation_foo
        ])] * (self.num_layers - 1)

        self.output_layers = [
            nn.Linear(in_features=self.hidden_size,
                      out_features=self.output_dim)
        ]

        self.sequential_layers = nn.Sequential(
            *(self.input_layers +
              self.hidden_layers +
              self.output_layers)
        )

    def forward(self, input_T):
        
        out_tensor = self.sequential_layers(input_T)
        return out_tensor

class MultisampleMLPDecoder(ArbitraryMLPDecoder):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_layers: int,
                 hidden_size: int,
                 activation_foo = nn.LeakyReLU(0.1),
                 ):
        super(MultisampleMLPDecoder, self).__init__(
            input_dim, output_dim, num_layers, hidden_size, activation_foo
        )

    def forward(self, input_t):
        
        batch_size, input_dim_local, num_samples = input_t.shape

        
        reshaped_t = input_t.transpose(1,2) 
        reshaped_t = reshaped_t.reshape([batch_size * num_samples, input_dim_local]) 
        encoded_t = super().forward(reshaped_t)
        return encoded_t

class MLPDecoderNeoEBM(ArbitraryMLPDecoder):
    def __init__(self,
                 y_dim: int,
                 predictor_code_dim: int = 8,
                 decoder_hidden_dim: int = 32,
                 batch_samples: bool = False,
                 num_layers: int = 3,
                 activation_foo=nn.LeakyReLU(0.1)
                 ):

        super(ArbitraryMLPDecoder, self).__init__()


class DecoderEBM(nn.Module, AbstractTimeSeriesEBMDecoderMixin):
    def __init__(self,
                 y_dim: int, 
                 predictor_code_dim: int = 8,
                 decoder_hidden_dim: int = 32,
                 batch_samples: bool = False,
                 num_layers: int = 3,
                 activation_foo = nn.LeakyReLU(0.1)
                 ):

        super(DecoderEBM, self).__init__()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



class LSTMNeoEBM(AbstractTimeseriesEBMV2):

    def __init__(self,
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
        
        super(LSTMNeoEBM, self).__init__(uses_flat_xs=False)

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

        
        self.c = nn.Parameter(torch.tensor([1.], requires_grad=use_normalizing_constant))

        self.x_encoder = PredictorEBM(x_dim=self.x_dim,
                                      lstm_hidden_size=self.lstm_hidden_size,
                                      predictor_code_dim=x_encoder_code_dim,
                                      lstm_num_layers=x_encoder_lstm_num_layers
                                      )
        self.x_decoder = ArbitraryMLPDecoder(
            input_dim=self.predictor_code_dim,
            output_dim=self.y_dim,
            num_layers=self.x_decoder_num_layers,
            hidden_size=self.x_decoder_code_dim,
        )
        self.y_encoder = MultisampleMLPDecoder(
            input_dim=self.y_dim,
            output_dim=self.predictor_code_dim,
            num_layers=self.y_decoder_num_layers,
            hidden_size=self.y_decoder_code_dim,
        )
        
        
        
        
        
        
        
        self.xy_decoder = ArbitraryMLPDecoder(
            input_dim=self.x_decoder_code_dim + self.y_decoder_code_dim,
            output_dim=1, 
            num_layers=self.decoder_num_layers,
            hidden_size=self.decoder_hidden_dim,
        )

        
        self.predictor = self.x_encoder
        self.decoder = self.xy_decoder
        self.feature_net = self.predictor
        self.predictor_net = self._get_decoded

    def get_decoded(self, xs: torch.Tensor, ys: torch.Tensor):
        encoded_x = self.get_encoded_x(xs) 
        return self._get_decoded(encoded_x, ys)

    def _get_decoded(self, encoded_x: torch.Tensor, ys: torch.Tensor):
        encoded_y = self.get_encoded_y(ys) 

        y_batch_size, input_dim_local, num_samples = ys.shape

        if (self.batch_samples):
            x_batch_size, _ = encoded_x.shape
            encoded_x_reshaped = encoded_x.unsqueeze(0).expand([y_batch_size // x_batch_size, -1, -1])\
                .reshape(y_batch_size, -1)
        else:
            encoded_x_reshaped = encoded_x.unsqueeze(1).expand(-1, num_samples, -1)  
            encoded_x_reshaped = encoded_x_reshaped.reshape(y_batch_size * num_samples, -1)  

        xy_encoded = torch.cat([encoded_x_reshaped, encoded_y], dim=1) 
        
        score = self.xy_decoder(xy_encoded)

        score = score.view(y_batch_size, num_samples) 
        return score

    def forward(self, xs: torch.Tensor, ys: torch.Tensor):
        return self.get_decoded(xs, ys)

    def get_encoded_x(self, xs: torch.Tensor) -> torch.Tensor:
        return self.x_encoder(xs)

    def get_encoded_y(self, ys: torch.Tensor) -> torch.Tensor:
        return self.y_encoder(ys)

    def get_decoded_y(self, encoded_xs_or_ys: torch.Tensor) -> torch.Tensor:
        return self.x_decoder(encoded_xs_or_ys)

    def get_decoded_xy(self, xs: torch.Tensor, ys: torch.Tensor):
        return self.get_decoded(xs, ys)

if __name__ == "__main__":
    test_neo_ebm = LSTMNeoEBM(
        x_dim=8,
        y_dim=2,
    )

    batch = 13
    input_dim = 3
    output_dim = input_dim + 1

    sample_count = 7
    input_for_multisample_decoder = torch.randn(
        [batch, input_dim, sample_count]
    )

    multisample_decoder = MultisampleMLPDecoder(input_dim=input_dim,
                          output_dim=output_dim,
                          num_layers=1,
                          hidden_size=5,
                          )
    multisample_output = multisample_decoder(input_for_multisample_decoder)

    

    lstm_neo_ebm = LSTMNeoEBM(
        x_dim=input_dim,
        y_dim=output_dim,
    )

    memory = 16
    sequential_x_for_lstm_neo_ebm = torch.randn(
        [batch, memory, input_dim]
    )
    y_for_lstm_neo_ebm = torch.randn(
        [batch, output_dim, sample_count]
    )
    energy_lstm_neo_ebm = lstm_neo_ebm(
        sequential_x_for_lstm_neo_ebm,
        y_for_lstm_neo_ebm
    )


