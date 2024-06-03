from torch.gnn_geometric.nn import MessagePassing
from torch_geometric.nn import knn_graph
from torch import cdist, index_select

from .layers import GravNetLayers

class GravNetLayer(nn.Module):

    def __init__(self, input_dim: int,
                       latent_output_dim: int,
                       feature_output_dim: int,
                       n_neighbors: int):

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

    def forward(self, x):

        # get the learned features and sptial coordinates
        learned_features = self.feature_network(x)
        spatial_coords   = self.latent_network(x)

        # generate edge index
        edge_index = knn_graph(spatial_coords, self.n_neighbors)

        # calculate the distance between neighbors
        neighbors = index_select(spatial_coords, 0, edge_index[1])
        distances = cdist(spatial, neighbors, metric='euclidian')
        weights   = self.kernel(distances)

        # used learned features for message passing between vertices
        messages = [x]
        for messanger in self.messangers:
            messages.append(messanger[learned,weights])

        all_features = torch.cat(messages, dim=1)

        return self.gravnet_layer(x)


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
            - layers (object): An object class containing dictionary with a following structure: { "input_dense": (), "n_aggr": int, "output_dense": ()}, where "input_dense" is a tuple indicating the numbers of nodes in each layer of the first dense network, the "output_dense" is a similary array for the output layer and  the "n_aggr" is the number of aggregation layers, "type_aggregation" is the aggregation method to be used in the graph, 
            - n_neighbors (int): Number of message passing neighbors for each data point.
            - kernel (str): kernel type, either "gaussian" or "exponential"
        """
        
        self.n_neighbors = n_neighbors
        self.layers      = layers

        if kernel not in _allowed_kernels:
            self.kernel      = _allowed_kernels[kernel]

        else:
            self.kernel      = None

        GravNetBlock()
    
    def buildInputLayer(self):
        """
        Function to build the input network components in the GravNet model.
        """

        # Get layers from the input_dense (x,...,y) such that y = n_latent_space + n_latent_features

        assert layers['input_layer'][-1] == (layers['n_latent_space'] + layers['n_latent_features']), "Dimension of the the input_dense should be compatible with the latent space!"

        input_block = []
        
        for ilayer, n_nodes in enumerate(self.layers['input_dense']):i
            n_next_nodes = self.layers['input_dense'][ilayer+1]
            if ilayer==len(self.layers['input_dense'])-1:
                input_block += nn.BatchNorm1d(n_nodes)
            else:
                input_block += nn.Linear(in_features=n_nodes, out_feature=n_next_nodes)
                input_block += nn.Tanh()

        input_block += 

        return nn.Sequential(_input_block)

    

    def build





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



