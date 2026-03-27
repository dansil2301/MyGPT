from NoTorchAI.GlobalState.Device import Device


class Quant:
    ALLOWED = [64, 32, 16]
    instance = None

    def __new__(cls, *args, **kwargs):
        if Quant.instance is None:
            Quant.instance = super().__new__(cls)
        return Quant.instance

    def __init__(self, quant: int = 32):
        if quant not in self.ALLOWED:
            raise ValueError(f"Unsupported quant: {quant}")
        self.quant = quant
        self.xp = Device().set_module()

    def set_quant(self):
        quant_obj = None
        if self.quant == 64:
            quant_obj = self.xp.float64
        elif self.quant == 32:
            quant_obj = self.xp.float32
        elif self.quant == 16:
            quant_obj = self.xp.float16
        else:
            raise ValueError("There is no such quant available")
        return quant_obj
