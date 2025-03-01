from energy.AbstractEnergyModels import *

class BabyEBM(nn.Module):
    def __init__(self, dim=2,
                 use_constant = True):
        super(BabyEBM, self).__init__()
        
        if (use_constant):
            self.c = nn.Parameter(torch.tensor([1.], requires_grad=True))
        else:
            self.c = torch.tensor(0.0)

        self.f = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            )

    def forward(self, x):
        log_p = - self.f(x) - self.c
        return log_p




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
        self.y_dim = y_dim
        self.predictor_code_dim = predictor_code_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.batch_samples = batch_samples
        self.num_layers = num_layers

        self.y_encoder = nn.Linear(self.y_dim, decoder_hidden_dim)

        self.xy_decoder_inputs = [
            nn.Linear(predictor_code_dim + decoder_hidden_dim, decoder_hidden_dim),
            activation_foo,
        ]
        self.xy_decoder_hiddens = [ResNetSequential([
            nn.Linear(in_features=decoder_hidden_dim,
                      out_features=decoder_hidden_dim),
            activation_foo
        ])] * (self.num_layers - 1)
        self.xy_decoder_outputs = [
            nn.Linear(decoder_hidden_dim, 1)
        ]

        self.sequential_layers = nn.Sequential(
            *(self.xy_decoder_inputs +
              self.xy_decoder_hiddens +
              self.xy_decoder_outputs)
        )
        self.sequential_layers_for_representation = nn.Sequential(
            *(self.xy_decoder_inputs + self.xy_decoder_hiddens)
        )

        
        
        
        
        

        
        
        
        
        
        
        

    def forward(self, x_encoded, y):
        
        

        if y.dim() == 2:
            y = y.view(-1,1) 

        y_batch_size, horizon, num_samples = y.shape

        x_feature = x_encoded 

        
        if (self.batch_samples):
            x_batch_size, _ = x_feature.shape
            x_feature = x_feature.unsqueeze(0).expand([y_batch_size // x_batch_size, -1, -1]).reshape(y_batch_size, -1)
        else:
            x_feature = x_feature.unsqueeze(1).expand(-1, num_samples, -1)  
            x_feature = x_feature.reshape(y_batch_size * num_samples, -1)  

        
        
        

        
        y_t = y.transpose(1,2) 
        y_t = y_t.reshape([y_batch_size * num_samples, horizon]) 

        y_encoded = self.y_encoder(y_t)
        
        xy_encoded = torch.cat([x_feature, y_encoded], dim=1) 

        
        
        
        
        
        

        score = self.sequential_layers(xy_encoded)

        
        score = score.view(y_batch_size, num_samples) 
        return score

    def get_final_representation(self, x_encoded, y):
        y_batch_size, horizon, num_samples = y.shape

        x_feature = x_encoded 

        
        if (self.batch_samples):
            x_batch_size, _ = x_feature.shape
            x_feature = x_feature.unsqueeze(0).expand([y_batch_size // x_batch_size, -1, -1]).reshape(y_batch_size, -1)
        else:
            x_feature = x_feature.unsqueeze(1).expand(-1, num_samples, -1)  
            x_feature = x_feature.reshape(y_batch_size * num_samples, -1)  


        
        y_t = y.transpose(1,2) 
        y_t = y_t.reshape([y_batch_size * num_samples, horizon]) 

        y_encoded = self.y_encoder(y_t)
        
        xy_encoded = torch.cat([x_feature, y_encoded], dim=1) 
        return self.sequential_layers_for_representation(xy_encoded)

class PredictorEBM(nn.Module, AbstractTimeSeriesEBMPredictorMixin):
    def __init__(self,
                 x_dim: int = 5,
                 lstm_hidden_size: int = 16,
                 predictor_code_dim: int = 16,
                 lstm_num_layers: int = 1,
                 ):

        super(PredictorEBM, self).__init__()
        self.x_dim = x_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.predictor_code_dim = predictor_code_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(input_size=self.x_dim,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers,
                            batch_first=True,
                            bidirectional=False,
                            dropout=0.0)
        self.linear = nn.Linear(self.lstm_hidden_size, self.predictor_code_dim)

    def forward(self, x):
        x_encoded, other = self.lstm(x)
        x_encoded = x_encoded[:,-1,:]
        x_encoded = self.linear(x_encoded)
        return x_encoded

class LSTMPredictorDecoderEBM(AbstractTimeseriesEBM):
    
    

    
    
    
    
    
    def __init__(self,
                 x_dim: int, 
                 y_dim: int,
                 lstm_hidden_size: int = 16,
                 predictor_code_dim: int = 8,
                 decoder_hidden_dim: int = 128,
                 lstm_num_layers: int = 1,
                 decoder_num_layers: int = 3,
                 use_normalizing_constant: bool = False,
                 batch_samples: bool = False,
                 ):

        super(LSTMPredictorDecoderEBM, self).__init__(uses_flat_xs=False)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.predictor_code_dim = predictor_code_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.batch_samples = batch_samples

        
        self.c = nn.Parameter(torch.tensor([1.], requires_grad=use_normalizing_constant))

        self.predictor = PredictorEBM(x_dim=self.x_dim,
                                      lstm_hidden_size=self.lstm_hidden_size,
                                      predictor_code_dim=predictor_code_dim,
                                      lstm_num_layers=lstm_num_layers
                                      )
        self.decoder = DecoderEBM(
            y_dim=self.y_dim,
            predictor_code_dim=self.predictor_code_dim,
            decoder_hidden_dim=self.decoder_hidden_dim,
            batch_samples=self.batch_samples,
            num_layers=decoder_num_layers,
        )

        
        self.feature_net = self.predictor
        self.predictor_net = self.decoder

    def get_encoded_x(self, x):
        return self.predictor(x)

    def get_decoded(self, x, y):
        x_encoded = self.get_encoded_x(x)
        decoded = self.decoder(x_encoded=x_encoded, y=y)
        return decoded

    def forward(self, x, y):
        log_p = self.get_decoded(x,y)
        return log_p

class MLPPredictorEBM(nn.Module, AbstractTimeSeriesEBMPredictorMixin):
    def __init__(self,
                 flattened_input_length: int,
                 hidden_size: int,
                 num_layers: int,
                 predictor_code_dim: int,
                 activation_foo = nn.LeakyReLU(0.1),
                 ):
        super(MLPPredictorEBM, self).__init__()
        self.input_length = flattened_input_length
        self.hidden_size = hidden_size
        self.predictor_code_dim = predictor_code_dim
        self.num_layers = num_layers

        if (self.num_layers < 1):
            raise ValueError(f"Number of layers '{self.num_layers} cannot be < 1")

        input_layers = [
            nn.Linear(in_features=self.input_length,
                      out_features=self.hidden_size),
            activation_foo,
        ]

        hidden_layers = [
            nn.Linear(in_features=self.hidden_size,
                      out_features=self.hidden_size),
            activation_foo,
        ] * (self.num_layers - 1)

        output_layers = [
            nn.Linear(in_features=self.hidden_size,
                      out_features=self.predictor_code_dim)
        ]

        self.sequential_layers = nn.Sequential(
            *(input_layers + hidden_layers + output_layers)
        )

    def forward(self, x):
        out_tensor = self.sequential_layers(x)
        return out_tensor

class ResNetSequential(torch.nn.Module):
    def __init__(self, list_of_modules):
        super().__init__()
        self.sequential_module = nn.Sequential(*list_of_modules)

    def forward(self, inputs):
        return self.sequential_module(inputs) + inputs

class MLPPredictorEBM_Residual(nn.Module, AbstractTimeSeriesEBMPredictorMixin):
    def __init__(self,
                 flattened_input_length: int,
                 hidden_size: int,
                 num_layers: int,
                 predictor_code_dim: int,
                 activation_foo = nn.LeakyReLU(0.1),
                 ):
        super(MLPPredictorEBM_Residual, self).__init__()
        self.input_length = flattened_input_length
        self.hidden_size = hidden_size
        self.predictor_code_dim = predictor_code_dim
        self.num_layers = num_layers

        if (self.num_layers < 1):
            raise ValueError(f"Number of layers '{self.num_layers} cannot be < 1")

        input_layers = [
            nn.Linear(in_features=self.input_length,
                      out_features=self.hidden_size),
            activation_foo
        ]

        hidden_layers = [ResNetSequential([
            nn.Linear(in_features=self.hidden_size,
                      out_features=self.hidden_size),
            activation_foo
        ])] * (self.num_layers - 1)

        output_layers = [
            nn.Linear(in_features=self.hidden_size,
                      out_features=self.predictor_code_dim)
        ]

        self.sequential_layers = nn.Sequential(
            *(input_layers + hidden_layers + output_layers)
        )

    def forward(self, x):
        out_tensor = self.sequential_layers(x)
        return out_tensor
















class MLPPredictorDecoderEBM(AbstractTimeseriesEBM):
    def __init__(self,
                 flattened_input_length: int,  
                 output_length: int,
                 predictor_hidden_size: int = 32,
                 decoder_hidden_dim: int = 128,
                 mlp_num_layers: int = 2,
                 decoder_num_layers: int = 3,
                 use_normalizing_constant: bool = False,
                 batch_samples: bool = False,
                 ):
        super(MLPPredictorDecoderEBM, self).__init__(uses_flat_xs=True)
        self.flattened_input_length = flattened_input_length
        self.output_length = output_length
        self.predictor_num_layers = mlp_num_layers
        self.predictor_hidden_size = predictor_hidden_size
        self.decoder_hidden_dim = decoder_hidden_dim
        self.batch_samples = batch_samples
        self.decoder_num_layers = decoder_num_layers

        
        self.c = nn.Parameter(torch.tensor([1.], requires_grad=use_normalizing_constant))

        self.predictor = MLPPredictorEBM(
            flattened_input_length=self.flattened_input_length,
            hidden_size=self.predictor_hidden_size,
            num_layers=self.predictor_num_layers,
            predictor_code_dim=predictor_hidden_size
        )

        self.decoder = DecoderEBM(
            y_dim=self.output_length,
            predictor_code_dim=self.predictor_hidden_size,
            decoder_hidden_dim=self.decoder_hidden_dim,
            batch_samples=self.batch_samples,
            num_layers=decoder_num_layers,
        )

        
        self.feature_net = self.predictor
        self.predictor_net = self.decoder

    def get_encoded_x(self, x):
        return self.predictor(x)

    def get_decoded(self, x, y):
        x_encoded = self.get_encoded_x(x)
        decoded = self.decoder(x_encoded=x_encoded, y=y)
        return decoded

    def forward(self, x, y):
        
        log_p = self.get_decoded(x, y)
        return log_p

class MLPPredictorDecoderEBM_residual(MLPPredictorDecoderEBM):
    def __init__(self,
                 flattened_input_length: int,  
                 output_length: int,
                 predictor_hidden_size: int = 32,
                 decoder_hidden_dim: int = 128,
                 mlp_num_layers: int = 1,
                 decoder_num_layers: int = 3,
                 use_normalizing_constant: bool = False,
                 batch_samples: bool = False,
                 ):
        super(MLPPredictorDecoderEBM_residual, self).__init__(
            flattened_input_length=flattened_input_length,
            output_length=output_length,
            predictor_hidden_size=predictor_hidden_size,
            decoder_hidden_dim=decoder_hidden_dim,
            mlp_num_layers=mlp_num_layers,
            use_normalizing_constant=use_normalizing_constant,
            batch_samples=batch_samples,
            decoder_num_layers=decoder_num_layers,
        )
        self.predictor = MLPPredictorEBM_Residual(
            flattened_input_length=self.flattened_input_length,
            hidden_size=self.predictor_hidden_size,
            num_layers=self.predictor_num_layers,
            predictor_code_dim=predictor_hidden_size
        )
