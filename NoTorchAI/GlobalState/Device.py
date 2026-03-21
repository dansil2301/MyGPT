class Device:
    ALLOWED = {"cpu": "numpy", "gpu": "cupy"}

    def __init__(self, device: str = "cpu"):
        if device not in self.ALLOWED:
            raise ValueError(f"Unsupported device: {device}")
        self.device = device

    def set_module(self):
        if self.device == "gpu":
            try:
                import cupy as xp
            except ImportError as e:
                raise ImportError("cupy required for gpu. install cupy.") from e
        else:
            import numpy as xp
        return xp
