class GravNetLayer(dict):
    """
    GravNetLayer is a data structure to be passed as an argument to GravNet class. The class supports item assignment only for the following list of keys: "input_dense", "output_dense" and "aggr"
    """
    def __init__(self):

        self.__attr      = ['input_dense', 'output_dense', 'aggr']

        self['input_dense']  = (4, 2, 1)
        self['output_dense'] = ()
    

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        """"""
        if key not in self.__attr:
            raise KeyError("Cannot assign %s to the class!" % key)
            return

        if key!='aggr':
            __item_check = all(type(item) is int for item in value)
            if not __item_check:
                raise TypeError('Item assignment for %s requires integer values!' % key)
                return

        else:
            __len = len(value)

            if __len!=2:
                raise ValueError('"Aggr" must be a tuple of type(n_aggr: int, aggr_type: str)!')
                return

            __item_check = type(value[0])==int and type(value[1]==str)
            if not __item_check:
                raise TypeError('"Aggr" must be a tuple of type(n_aggr: int, aggr_type: str)!')
                return

        setattr(self, key, value)
