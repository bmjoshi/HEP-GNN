import json

class GravNetMap(dict):
    """
    GravNetMap is a data structure to be passed as an argument to GravNet class.
    The network is a collection of several GravNetBloacks connected in series.
    """
    def __init__(self):
        """
        Input Parameters:
            - n_input_features  (int) : number of input features (default: 4)
            - n_hidden_layers   (int) : number of hidden layers within the input dense (default: 3)
            - n_hidden nodes    (int) : number of hidden nodes within the input dense (default: 64)
            - n_output_features (int) : number of output features (default: 3)
            - n_blocks          (int) : number of blocks used in GravNet (default: 3)
            - n_latent_space    (int) : number of features used in the latent space (default: 2)
            - n_latent_features (int) : number of features assigned to nodes in the latent space
            - n_neighbors       (int) : number of nearest neighbors for building graph
            - kernel            (str) : kernel for computing distance (default: 'gauss')
            - type_aggr         (list(str)): a list of aggregation type to be used for graph convolution
            - n_final_dense     (int) : number of nodes in the final layer
            - n_final_output    (int) : dimension of the final output
        """
        self.__attr = ['input_dense', 'output_dense', 'n_blocks', 'n_latent_space', 'n_latent_features']
        self.__dict = {}

        self.resetMap()
    
    def resetMap(self):
        """
        Reset map
        """
        self.__dict['n_input_features']    = 4
        self.__dict['n_hidden_layers']     = 3
        self.__dict['n_hidden_nodes']      = 64
        self.__dict['n_output_features']   = 2
        self.__dict['n_blocks']            = 3
        self.__dict['n_latent_space']      = 2
        self.__dict['n_latent_features']   = 3
        self.__dict['n_neighbors']         = 4
        self.__dict['kernel']              = 'gauss'
        self.__dict['type_aggr']           = ['sum','max']
        self.__dict['n_final_dense']       = 128
        self.__dict['n_final_output']      = 1

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
            assert type(value) == list, 'A list of aggregation modes is expected!'
            for item in value:
                assert type(value) == str, 'Item assignment for %s requires integer values!' % key

        else:
            assert type(value) == str, 'Item assignment for %s requires string values!' % key

        self.__dict[key] = value

    def getNetworkMap(self):
        tmpdict = {}
        for k in self.__dict.keys():
            tmpdict[k] = self.__dict[k]
        return tmpdict
    
    def setNetworkMap(self, configfile="") -> None:
        with open(configfile, 'r') as json_:
            config = json.load(json_)
        for k in config:
            self.__dict[k] = config[k]