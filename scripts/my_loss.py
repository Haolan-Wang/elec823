from utils import ErrorCal
import torch
import torch.nn as nn
from torchmetrics import SpearmanCorrCoef

class MyLoss_v0_1(nn.MSELoss):
    """Useing MSE loss only"""
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = ErrorCal(input, target)
        return error.mse_loss
    
class MyLoss_v0_2(nn.MSELoss):
    """Useing MAE loss only"""
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = ErrorCal(input, target)
        return error.mae_loss

class MyLoss_v1_1(nn.MSELoss):
    """Using pearson loss only"""
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = ErrorCal(input, target)
        pearson_loss = 1 - error.pearson_coef
        return pearson_loss
    

# class MyLoss_v1_2(nn.MSELoss):
#     """Using spearman loss only"""
#     __constants__ = ['reduction']
#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super().__init__(size_average, reduce, reduction)

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         error = ErrorCal(input, target)
#         spearman_coef = error.spearman_coef
#         spearman_loss = 1 - spearman_coef
#         print(spearman_loss.requires_grad)
#         return spearman_loss
    
class MyLoss_v2_1(nn.MSELoss):
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
    
class MyLoss_v2_2(nn.MSELoss):
    """Using MAE+pearson loss"""
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = ErrorCal(input, target)
        pearson_loss = 1 - error.pearson_coef
        mae_loss = error.mae_loss
        
        loss = mae_loss + pearson_loss
        return loss
    
# class MyLoss_v2_3(nn.MSELoss):
#     """Using MSE+spearman loss"""
#     __constants__ = ['reduction']
#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super().__init__(size_average, reduce, reduction)

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         error = ErrorCal(input, target)
#         spearman_loss = 1 - error.spearman_coef
#         mse_loss = error.mse_loss
        
#         loss = mse_loss + spearman_loss
#         return loss
    
# class MyLoss_v2_4(nn.MSELoss):
#     """Using MAE+spearman loss"""
#     __constants__ = ['reduction']
#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super().__init__(size_average, reduce, reduction)

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         error = ErrorCal(input, target)
#         spearman_loss = 1 - error.spearman_coef
#         mae_loss = error.mae_loss
        
#         loss = mae_loss + spearman_loss
#         return loss
    
    
# class ErrorCal2():
#     """Input y_true and y_pred when initilizing, return pearson, spearman, mse loss"""
#     def __init__(self, y_true, y_pred):
#         self.mse_loss = torch.nn.functional.mse_loss(y_true, y_pred)*y_true.shape[0]
#         self.pearson_coef = torch.corrcoef(torch.stack((y_true, y_pred)))[0, 1]*y_true.shape[0]
#         self.spearman = SpearmanCorrCoef()
#         self.spearman_coef = self.spearman(y_true, y_pred)*y_true.shape[0]

# class MyLoss_v2(nn.MSELoss):
#     __constants__ = ['reduction']
#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super().__init__(size_average, reduce, reduction)

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         error = ErrorCal2(input, target)
#         pearson_loss = 1 - error.pearson_coef
#         spearman_loss = 1 - error.spearman_coef
#         corr_loss = pearson_loss + spearman_loss
#         mse_loss = error.mse_loss
        
#         loss = mse_loss + corr_loss
        
#         return loss

# class MyLoss_v3(nn.MSELoss):
#     """
#         Pearson loss = 1 - pearson_coef only
#     """
#     __constants__ = ['reduction']
#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super().__init__(size_average, reduce, reduction)

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         error = ErrorCal(input, target)
#         pearson_loss = 1 - error.pearson_coef
        
#         return pearson_loss