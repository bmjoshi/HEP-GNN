import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import knn_graph
from torch import cdist

import numpy as np

from .kernels import *

class GravNetMessagePassing(MessagePassing):
    """
    The GravNetMessagePassing layer takes the features nodes, their edge indices and weights, which are
    assigned to the edges according to their distances in the lantent space, and passes messages to the nearest neighbors.
    The messages that are passed between the nodes are aggregated according to the methods specified by 
    the user through an input arguments. The aggregated messages are finally passed through a dense network to
    produce the final output.
    """
    def __init__(self, aggr=['add'], input_dim=3, output_dim=3):
        super().__init__(aggr=aggr)
        
        self.aggr         = aggr
        self.input_dim    = input_dim
        self.output_dim   = output_dim
        _output_dense     = []
        
        _output_dense.append(nn.Linear(in_features=input_dim*(len(self.aggr)+1), out_features=self.output_dim))
        self.output_dense = nn.Sequential(*_output_dense)

    def forward(self, x, edge_index, weights):
        
        # get the weights of the nodes by using gaussian kernel
        
        out = self.propagate(edge_index, x=x, weights=weights)
        out = torch.cat([x, out], dim=1)
        out = self.output_dense(out)
        
        return out

    def message(self, x_j, w):
        print(w, x_j)
        return w.view(-1,1)*x_j


class GravNetLayer(nn.Module):
    """
    GravNetLayer is a layer within the GravNetBlock. It takes input features and passes them
    simultaneously through a fully connected dense networks. One of them produces new features in
    a latent space and the other assigns spatial distances to them within the same space. Using those
    distances, it builds a KNN map and uses GravNetMessagePassing for running graph convolution.
    """
    def __init__(self, input_dim: int,
                       latent_output_dim: int,
                       feature_output_dim: int,
                       n_neighbors: int,
					   kernel: str,
					   type_aggr: list):
        super().__init__()
        
        self.input_dim          = input_dim
        self.latent_output_dim  = latent_output_dim
        self.feature_output_dim = feature_output_dim 
        self.n_neighbors        = n_neighbors

        _latent_network  = []
        _feature_network = []
        
        _latent_network.append(nn.Linear(in_features=input_dim, out_features=latent_output_dim))
        _latent_network.append(nn.Tanh())
        _feature_network.append(nn.Linear(in_features=input_dim, out_features=feature_output_dim))
        _feature_network.append(nn.Tanh())

        self.feature_network = nn.Sequential(*_feature_network)
        self.latent_network  = nn.Sequential(*_latent_network)

        self.messangers = GravNetMessagePassing(type_aggr)

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

        # calculate the distance between neighbors
        distances = cdist(spatial_coords, spatial_coords, p=2) # p=2 for euclidian distances
        distances = distances[edge_index[0], edge_index[1]]
        weights   = self.kernel(distances)

        # used learned features for message passing between vertices
        out = self.messanger(learned_features, edge_index, weights=weights)

        return out


class GravNetBlock(nn.Module):
    """
    A GravNetBlock consists of a input dense network followed by a GravNetLayer
    """
    def __init__(self, layers = (10,64,64,64),
                       n_latent_size = 4,
                       n_latent_features = 22,
                       n_features_out = 48,
                       n_neighbors = 4,
                       kernel = 'gauss',
                       type_aggr = ['mean']):

        super(GravNetBlock, self).__init__()

        _n_input_dim   = layers[0]+1
        _input_layer   = []
        _fout_layer    = []

        _n_prev = _n_input_dim

        for ilayer, layer_ in enumerate(layers):
            if ilayer==0: continue
            _input_layer.append(nn.Linear(in_features=_n_prev, out_features=layer_))
            _input_layer.append(nn.Tanh())
            _n_prev = layer_

        _gravnet_layer = [GravNetLayer(input_dim = layers[-1],
                                      latent_output_dim  = n_latent_features,
                                      feature_output_dim = n_latent_features,
                                      n_neighbors = n_neighbors,
                                      kernel = kernel,
                                      type_aggr = type_aggr)]

        _fout_layer.append(nn.Linear(in_features=n_latent_features, out_features=n_features_out))
        _fout_layer.append(nn.BatchNorm1d(n_features_out))

        self.block = nn.Sequential(*(_input_layer + _gravnet_layer + _fout_layer))
        _n_prev = _n_input_dim
    
    def forward(self, x):
        # begin with concatentating the input with its mean
        x = torch.cat([x, torch.mean(x, dim=1).view(-1,1)], dim=1)
        x = self.block(x)
 
        return x


class GravNet(nn.Module):
    """
    Implementation of GravNet model described in https://link.springer.com/article/10.1140/epjc/s10052-019-7113-9#Tab2.
    The model uses several GNN-based blocks series, each of which transofrm given input features 
    into another set of features in a latent space. The final block output is then passed through a 
    fully-connected dense layer to get the desired output.
    """

    def __init__(self,
                 network_map: object) -> None:
        super(GravNet, self).__init__()
        """
        Input Parameters:
            - network_map (object): NetworkMap object
            - kernel (str): kernel type, either "gaussian" or "exponential"
        """
        
        self.network_map  = network_map
        
        _network     = []
        __input_dim  = network_map['n_input_features']
        
        # build network blocks
        for iblock in range(self.network_map['n_blocks']):
            __input_layer  = [__input_dim]
            for ihid in range(self.network_map['n_hidden_nodes']):
                __input_layer += [self.network_map['n_hidden_nodes']]
            __input_layer += [self.network_map['n_output_features']]
            _network.append(GravNetBlock(layers=tuple(__input_layer),
                                            n_latent_size = self.network_map['n_latent_space'],
                                            n_latent_features = self.network_map['n_latent_features'],
                                            n_features_out = self.network_map['n_output_features'],
                                            n_neighbors = self.network_map['n_neighbors'],
                                            kernel = self.network_map['kernel'],
                                            type_aggr = self.network_map['type_aggr']))
            
            __input_dim   = self.network_map['n_output_features']
        
        _network.append(nn.Linear(in_features  = self.network_map['n_output_features'],
                                          out_features = self.network_map['n_final_dense']))
        _network.append(nn.ReLU())
        _network.append(nn.Linear(in_features = self.network_map['n_final_dense'],
                                   out_features = self.network_map['n_final_output']))
        self.network = nn.Sequential(*_network)
    
    def forward(self, x):
        return self.network(x)