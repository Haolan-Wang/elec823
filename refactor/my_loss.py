from utils import ErrorCal
import torch
import torch.nn as nn
from torchmetrics import SpearmanCorrCoef


class MyMSELoss(nn.MSELoss):
    """Useing MSE loss only"""
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = ErrorCal(input, target)
        return error.mse_loss


class MSEPearsonLoss(nn.MSELoss):
    """Using MSE+pearson loss: so far the best one"""
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = ErrorCal(input, target)
        pearson_loss = 1 - error.pearson_coef
        mse_loss = error.mse_loss

        loss = mse_loss + pearson_loss
        return loss
    
class MSEConcordanceLoss(nn.MSELoss):
    """Using MSE+CCC loss: so far the best one"""
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = ErrorCal(input, target)
        concordance_loss = 1 - error.ccc
        mse_loss = error.mse_loss

        loss = mse_loss + concordance_loss
        return loss

class MSEPearsonConcordanceLoss(nn.MSELoss):
    """Using MSE+CCC loss: so far the best one"""
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = ErrorCal(input, target)
        concordance_loss = 1 - error.ccc
        mse_loss = error.mse_loss
        pearson_loss = 1 - error.pearson_coef

        loss = mse_loss + concordance_loss + pearson_loss
        return loss

# class ConcordanceLoss(nn.MSELoss):
#     """Using CCC loss: so far the best one"""
#     __constants__ = ['reduction']

#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super().__init__(size_average, reduce, reduction)

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         error = ErrorCal(input, target)
#         concordance_loss = 1 - error.ccc

#         loss = concordance_loss
#         return loss