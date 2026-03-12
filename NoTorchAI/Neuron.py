import pprint as pp


class Neuron:
    def _save(self, layer_id: int = 0, saved: dict = {}):
        """
        I made this saving mechanism depend on the order of attributes 
        initialized in the __init__ (not the most perfect lol)
        """
        # todo: rebuild to create a folded list with all subclasses (easier to manage)
        c_cls_name = self.__class__.__name__
        c_layer_id = layer_id
        
        neuron_id = f"{c_layer_id}_{c_cls_name}"
        saved[neuron_id] = {}

        for attr_name in reversed(self.__dict__):
            if isinstance(self.__dict__[attr_name], Neuron):
                layer_id += 1
                saved, layer_id = self.__dict__[attr_name]._save(layer_id, saved)
            elif isinstance(self.__dict__[attr_name], list):
                ls = self.__dict__[attr_name]
                if ls and isinstance(ls[0], Neuron):
                    for neuron in ls:
                        layer_id += 1
                        saved, layer_id = neuron._save(layer_id, saved)
            else:
                saved[neuron_id][attr_name] = self.__dict__[attr_name]
        
        print()
        print(pp.pformat(saved))
        return saved, layer_id


        pass

    def save(self, path: str) -> None:
        self._save()
        pass

    def load(self, path: str) -> None:
        pass
