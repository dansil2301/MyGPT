import numpy as np


class Neuron:
    def _save(self, saved: dict = {}):
        """
        Saves information in the folded list
        """
        c_cls_name = self.__class__.__name__
        saved[c_cls_name] = {}

        for attr_name in self.__dict__:
            if isinstance(self.__dict__[attr_name], Neuron):
                saved[c_cls_name][attr_name] = {}
                saved[c_cls_name][attr_name] = self.__dict__[attr_name]._save(saved[c_cls_name][attr_name])
            elif isinstance(self.__dict__[attr_name], list):
                ls = self.__dict__[attr_name]
                if ls and isinstance(ls[0], Neuron):
                    for neuron in ls:
                        if attr_name not in saved[c_cls_name]:
                            saved[c_cls_name][attr_name] = []
                        saved[c_cls_name][attr_name].append(neuron._save({}))
            else:
                saved[c_cls_name][attr_name] = self.__dict__[attr_name]

        return saved

    def save(self, path: str) -> None:
        saved = self._save()
        np.savez(path + ".npz", saved)

    def load(self, path: str) -> None:
        saved = np.load(path + ".npz", allow_pickle=True)
        print(saved.files)
        for key in saved.files:
            print(saved[key])
