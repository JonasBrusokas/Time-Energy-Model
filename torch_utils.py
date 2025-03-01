import numpy as np
from sklearn.preprocessing import StandardScaler
from torch import nn


def set_grad_flow_for_nn(model: nn.Module, enable_grad_flow: bool):
    """
    Sets gradient flow for a given nn.Module. Allows "(un-)freezing".
    :param model: nn.Module
    :param enable_grad_flow: True/False
    :return: None
    """
    if (model is not None) and (isinstance(model, nn.Module)):
        for param in model.parameters():
            param.requires_grad = enable_grad_flow


def inverse_scale_std_scaler_values(
    std_scaler: StandardScaler, values: np.array, column_index: int
):
    """
    :param std_scaler: fitted scaler
    :param values: numpy array of values to be inverse transformed
    :param column_index: column index of n_features of the fitted scaler
    :return: numpy array of transformed values
    """
    return values * std_scaler.scale_[column_index] + std_scaler.mean_[column_index]


def count_parameters(model):
    """
    :param model: typical 'torch' model
    :return: integer of the parameter count of the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Metrics:
    @staticmethod
    def MAE(pred, true):
        return np.mean(np.abs(pred - true))

    @staticmethod
    def MSE(pred, true):
        return np.mean((pred - true) ** 2)

    @staticmethod
    def RMSE(pred, true):
        return np.sqrt(Metrics.MSE(pred, true))

    @staticmethod
    def MAPE(pred, true):
        return np.mean(np.abs((pred - true) / true))

    @staticmethod
    def mape(pred, true):
        
        mask = true != 0

        
        mape = np.abs((true - pred) / true)
        mape[~mask] = np.nan

        
        return np.nanmean(mape)

    @staticmethod
    def smape(pred, true):
        

        
        denominator = (np.abs(true) + np.abs(pred)) / 2.0
        numerator = np.abs(pred - true)
        smape = numerator / denominator

        
        smape[denominator == 0] = 0.0

        
        return np.nanmean(smape)
