from torch_geometric.nn import MessagePassing
from torch_geometric.nn import knn_graph
from torch import cdist, index_select

import numpy as np

from .kernels import *
from .networkMap import GravNetMap

class GravNetMessagePassing(MessagePassing):
    def __init__(self, aggr=['add'], input_dim=3, output_dim=3):
        super().__init__(aggr=aggr)
        
        self.aggr         = aggr
        self.input_dim    = input_dim
        self.output_dim   = output_dim
        self.output_dense = nn.Sequential()
        
		self.output_dense.append(nn.Linear(in_features=input_dim*(len(self.aggr)+1), out_features=self.output_dim))


    def forward(self, x, edge_index, weights=w):
        
        # get the weights of the nodes by using gaussian kernel
        
        out = self.propagate(edge_index, x=x, weights=w)
        out = torch.cat([x, out], dim=1)
		out = self.output_dense(out)
        
		return out

    def message(self, x_j, w):
        print(w, x_j)
        return w.view(-1,1)*x_j


class GravNetLayer(nn.Module):

    def __init__(self, input_dim: int,
                       latent_output_dim: int,
                       feature_output_dim: int,
                       n_neighbors: int,
					   kernel: str,
					   aggr: list):

        self.latent_output_dim  = latent_output_dim
        self.feature_output_dim = feature_output_dim 

        self.latent_network  = []
        self.feature_network = [] 

        self.latent_network += nn.Linear(in_features=input_dim, out_features=latent_output_dim)
        self.latent_network += nn.Tanh()
        self.latent_network  = nn.Sequential(self.latent_network)

        self.feature_network += nn.Linear(in_features=input_dim, out_features=feature_output_dim)
        self.feature_network += nn.Tanh()
        self.feature_network  = nn.Sequential(self.feature_network)

        self.messangers = GravNetMessagePassing(aggr)

        # TODO: explore and add more kernels
        if kernel not in kernel_list:
            raise ValueError('Kernel %s not in the specified list!'%kernel)

        self.kernel       = kernel_map[kernel]


    def forward(self, x):

        # get the learned features and sptial coordinates
        learned_features = self.feature_network(x)
        spatial_coords   = self.latent_network(x)

        # generate edge index
        edge_index = knn_graph(spatial_coords, self.n_neighbors)
		
		self.messanger(aggr=self.aggr, input_dim=self.feature_output_dim, output_dim=self.latent_output_dim)


        # calculate the distance between neighbors
        distances = cdist(spatial_coords, spatial_coords, p=2) # p=2 for euclidian distances
        distances = distances[edge_index[0], edge_index[1]]
		weights   = self.kernel(distances) 

        # used learned features for message passing between vertices
		out = self.messanger(features, edge_index, weights=weights)

        return out


class GravNetBlock(nn.Module):

    def __init__(self, layer_map=(10,64,64,64),
                       n_latent_size=4,
                       n_latent_features=22,
                       n_features_out=48):

        super(GravNetBlock, self).__init__()

        self.n_input_dim       = layer_map[0]+1
        self.layer_map         = layer_map
        self.n_latent_size     = n_latent_size
        self.n_latent_features = n_latent_features
        self.n_features_out    = n_features_out

        _input_layer   = []
        _gravnet_layer = []
        _fout_layer    = []

        _n_prev = n_input_dim
        _n_next = n_hidden_dim

        for ilayer, layer_ in enumerate(self.layer_map):
            _input_layer += nn.Linear(in_features=_n_prev, out_features=layer_)
            _input_layer += nn.Tanh()
            _n_prev = _n_next

        _gravnet_layer += GravNetLayer(input_dim=layer_map[-1],
                                      latent_output_dim = n_latent_space,
                                      feature_output_dim = n_latent_features)

        _fout_layer += nn.Linear(in_features=n_latent_features, out_features=n_features_out)

        _fout_layer += nn.BatchNorm1d(n_features_out)

        self.block = nn.Sequential(_input_layer + _gravnet_layer + _fout_layer)

    def forward(self, x):

        # begin with concatentating the input with its mean
        x = torch.cat([x, torch.mean(x)], dim=1)
        x = self.block(x)

        return x


class GravNet(nn.Module):
    """
    The GravNet model transforms V input features into another feature F in a latent space of S dimensions. The nodes in the latent space will aggregate feature information from a select nearest neighbors and update its own feature value. After several aggregations, all the updated values of the features will be passed through another fully-connected dense network to obtain the desired output.
    """

    def __init__(self,
                 layers: object
                 n_neighbors: int,) -> None:
        super(GravNet, self).__init__()
        """
        Input Parameters:
            - layers (object): An object class containing dictionary with a following structure: { "input_dense": (), "n_blocks": int, "output_dense": ()}, where "input_dense" is a tuple indicating the numbers of nodes in each layer of the first dense network, the "output_dense" is a similary array for the output layer and  the "n_blocks" is the number of aggregation layers, "type_aggregation" is the aggregation method to be used in the graph, 
            - n_neighbors (int): Number of message passing neighbors for each data point.
            - kernel (str): kernel type, either "gaussian" or "exponential"
        """
        
        self.n_neighbors = n_neighbors
        self.layers      = layers
        self.n_blocks    = n_blocks
        self.network     = nn.Sequential()

        self.network.append(GravNetBlock)

        self.block = GravNetBlock()




    def buildNetwork(self):
        """
        Function to add network components together to build GravNet.
        """

        # make input block
        self.input_layer = 
        

    def layerMap(self) -> object:
        """
        Function to return the network map for each layers of the GravNet
        """
        # check if the input layer is inilialized
        self.layers['input_dense']



    def forward(self, x, batch=None):

        spatial, features = self.first_dense(x)
        
        edge_index        = knn_graph(spatial, self.n_neighbors, batch, loop=False)



