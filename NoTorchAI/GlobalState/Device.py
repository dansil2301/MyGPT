class Device:
    ALLOWED = {"cpu": "numpy", "gpu": "cupy"}
    instance = None

    def __new__(cls, *args, **kwargs):
        if Device.instance is None:
            Device.instance = super().__new__(cls)
        return Device.instance

    def __init__(self, device: str = "gpu"):
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
