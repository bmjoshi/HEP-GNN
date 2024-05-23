class GravNetMap(dict):
    """
    GravNetMap is a data structure to be passed as an argument to GravNet class. The class supports item assignment only for the following list of keys: "input_dense", "output_dense" and "aggr"
    """
    def __init__(self):

        self.__attr = ['input_dense', 'output_dense', 'n_aggr', 'n_latent_space', 'n_latent_features']
        self.__dict = {}

        self.__dict['input_dense']       = (4, 2, 1)
        self.__dict['output_dense']      = (4, 2, 1)
        self.__dict['n_aggr']            = 3
        self.__dict['type_aggr']         = 'sum'
        self.__dict['n_latent_space']    = 2
        self.__dict['n_latent_features'] = 3

    def __getitem__(self, key):
        return self.__dict[key]

    def __setitem__(self, key: str, value) -> None:
        """"""
        if key not in self.__attr:
            raise KeyError("Cannot assign %s to the class!" % key)
            return

        if 'n_' in key:
            assert type(value) == int, 'Item assignment for %s requires integer values!' % key

        elif 'type_' in key:
            assert type(value) == str

        else:
            for item in value:
                assert type(item) == int, 'Item assignment for %s requires integer values!' % key

        self.__dict[key] = value

    def getLayers(self):
        tmpdict = {}
        for k in self.__dict.keys():
            tmpdict[k] = self.__dict[k]
        return tmpdict
