import cupy as cp
import numpy as np

from NoTorchAI.GlobalState.Device import Device
from NoTorchAI.GlobalState.Quant import Quant


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
    
    def _find_implementation(self, name: str) -> type:
        try:
            subclasses = Neuron.__subclasses__()
            for subclass in subclasses:
                if subclass.__name__ == str(name):
                    return subclass
        except:
            return None
        return None
    
    def _check_dict_content_for_class_name(self, saved_dic: dict) -> object:
        class_name = list(saved_dic.keys())[0]
        attrs = saved_dic[class_name]
        impl_class = self._find_implementation(class_name)
        if impl_class:
            instance = impl_class.__new__(impl_class)
            self.set_xp_value(instance)
            self._load(attrs, instance)
            return instance
        else:
            return None

    def set_xp_value(self, instance: object):
        instance.device_str = 'cpu'
        instance.device = Device('cpu')
        instance.xp = instance.device.set_module()

    def _load(self, saved: dict, c_class_impl: object = None):
        for el in saved:
            # check if the dict is an element or a folded Neuron
            if isinstance(saved[el], dict):
                instance = self._check_dict_content_for_class_name(saved[el])
                if instance:
                    setattr(c_class_impl, el, instance)
                else:
                    setattr(c_class_impl, el, saved[el])

            # check if array element is an element of a folded Neuron
            elif isinstance(saved[el], list):
                new_ls = []
                for ls_el in saved[el]:
                    instance = None
                    if isinstance(ls_el, dict):
                        instance = self._check_dict_content_for_class_name(ls_el)
                    if instance:
                        new_ls.append(instance)
                    else:
                        new_ls.append(ls_el)
                setattr(c_class_impl, el, new_ls)

            # if not Neuron just save the value to the variable
            else:
                setattr(c_class_impl, el, saved[el])
        
        return c_class_impl

    def save(self, path: str) -> None:
        saved = self._save()
        np.savez(path + ".npz", arr_0=saved)

    def load(self, path: str):
        archive = np.load(path + ".npz", allow_pickle=True)

        saved = archive["arr_0"]
        if isinstance(saved, np.ndarray) and saved.dtype == object:
            saved = saved.item()

        # create instance of the saved class and load into it
        init_name = list(saved.keys())[0]
        init_class = self._find_implementation(init_name)
        instance = init_class.__new__(init_class)
        self._load(saved[init_name], instance)
        
        # set device and xp
        self.set_xp_value(instance)
        
        return instance
