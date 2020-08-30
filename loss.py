"""
A loss function measure how godo our prediction are.
We can use this to adjsut the parameter of out neural net.
"""

from tensor import Tensor
import numpy as np

class Loss:
    def loss(self, predicted: Tensor, actual:Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actial: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):

    """
    Implemeting mean squared loss, 
    but building actual loss for now
    """
    def loss(self, predicted: Tensor, actual:Tensor) -> float:
        return np.sum((predicted - actual)**2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2*(predicted - actual)
