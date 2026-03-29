from typing import Any, List

import cupy as cp
import numpy as np

from NoTorchAI.GlobalState.Device import Device
from NoTorchAI.GlobalState.Quant import Quant


_xp = Device().set_module()
_quant = Quant().set_quant()


class MatrixOperations:
    xp = _xp
    quant = _quant

    pi: float = float(np.pi)
    inf: float = float(np.inf)

    @classmethod
    def xavier_init(cls, x: int, y: int) -> np.ndarray | cp.ndarray:
        std = 1.0 / (y ** 0.5)
        return cls.xp.random.normal(0, std, (x, y)).astype(cls.quant)

    @classmethod
    def he_normal_init(cls, x: int, y: int) -> np.ndarray | cp.ndarray:
        std = (2.0 / x) ** 0.5
        return cls.xp.random.normal(0, std, (x, y)).astype(cls.quant)
    
    @classmethod
    def ones(cls, x: int, dtype: type = None) -> np.ndarray | cp.ndarray:
        if dtype:
            return cls.xp.ones(x, dtype=dtype)
        return cls.xp.ones(x, dtype=cls.quant)
    
    @classmethod
    def zeros(cls, x: int) -> np.ndarray | cp.ndarray:
        return cls.xp.zeros(x, dtype=cls.quant)
    
    @classmethod
    def zeros_like(cls, x: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        return cls.xp.zeros_like(x)
    
    @classmethod
    def linalg_norm(cls, x: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        return cls.xp.linalg.norm(x)
    
    @classmethod
    def sum(cls, x: np.ndarray | cp.ndarray, axis: int | tuple = None, keepdims: bool = None) -> np.ndarray | cp.ndarray:
        return cls.xp.sum(x, axis=axis, keepdims=keepdims)
    
    @classmethod
    def max(cls, x: np.ndarray | cp.ndarray, axis: int | tuple = None, keepdims: bool = None) -> np.ndarray | cp.ndarray:
        return cls.xp.max(x, axis=axis, keepdims=keepdims)
    
    @classmethod
    def exp(cls, x: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        return cls.xp.exp(x)
    
    @classmethod
    def log(cls, x: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        return cls.xp.log(x)

    @classmethod
    def argmax(cls, x: np.ndarray | cp.ndarray, axis: int | tuple = None, keepdims: bool = False) -> np.ndarray | cp.ndarray:
        return cls.xp.argmax(x, axis=axis, keepdims=keepdims)

    @classmethod
    def expand_dims(cls, x: np.ndarray | cp.ndarray, axis: int) -> np.ndarray | cp.ndarray:
        return cls.xp.expand_dims(x, axis=axis)
    
    @classmethod
    def add_at(cls, x, y, z) -> None:
        cls.xp.add.at(x, y, z)
    
    @classmethod
    def sqrt(cls, x: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        return cls.xp.sqrt(x)
    
    @classmethod
    def mean(cls, x: np.ndarray | cp.ndarray, axis: int | tuple = None, keepdims: bool = None) -> np.ndarray | cp.ndarray:
        return cls.xp.mean(x, axis=axis, keepdims=keepdims)
    
    @classmethod
    def var(cls, x: np.ndarray | cp.ndarray, axis: int | tuple = None, keepdims: bool = None) -> np.ndarray | cp.ndarray:
        return cls.xp.var(x, axis=axis, keepdims=keepdims)
    
    @classmethod
    def cos(cls, x: np.ndarray | cp.ndarray):
        return cls.xp.cos(x)
    
    @classmethod
    def where(cls, mask: np.ndarray | cp.ndarray, x: np.ndarray | cp.ndarray, choice: Any):
        return cls.xp.where(mask, x, choice)
    
    @classmethod
    def clip(cls, x: np.ndarray | cp.ndarray, minimum: float, maximum: float) -> np.ndarray | cp.ndarray:
        return cls.xp.clip(x, minimum, maximum)
    
    @classmethod
    def triu(cls, x: np.ndarray | cp.ndarray, k: int = 0) -> np.ndarray | cp.ndarray:
        return cls.xp.triu(x, k=k)

    @classmethod
    def matmul(cls, x: np.ndarray | cp.ndarray, y: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        return cls.xp.matmul(x, y)

    @classmethod
    def arange(cls, x: int) -> np.ndarray | cp.ndarray:
        return cls.xp.arange(x)
    
    @classmethod
    def concatenate(cls, arrays: list[np.ndarray | cp.ndarray], axis: int = 0) -> np.ndarray | cp.ndarray:
        return cls.xp.concatenate(arrays, axis=axis)
    
    @classmethod
    def copy(cls, x: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        return cls.xp.copy(x)
    
    @classmethod
    def convert_to_xp(cls, array: List[Any]) -> np.ndarray | cp.ndarray:
        return cls.xp.array(array)
    
    @classmethod
    def get_list(cls, array: np.ndarray | cp.ndarray) -> list:
        if isinstance(array, cp.ndarray):
            array = array.get()
        return list(array)
