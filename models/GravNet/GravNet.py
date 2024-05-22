from torch.gnn_geometric.nn import MessagePassing
from torch_geometric.nn import knn_graph
from torch import cdist, index_select

from .layers import GravNetLayer

class GravNet(nn.Module):
    """
    The GravNet model transforms V input features into another feature F in a latent space of S dimensions. The nodes in the latent space will aggregate feature information from a select nearest neighbors and update its own feature value. After several aggregations, all the updated values of the features will be passed through another fully-connected dense network to obtain the desired output.
    """

    def __init__(self,
                 n_layers: object
                 n_neighbors: int,) -> None:
        """
        Input Parameters:
            - n_layers (object): An object class containing dictionary with a following structure: { "input_dense": (), "aggr": (n_aggr, aggr_type), "output_dense": ()}, where "input_dense" is a tuple indicating the numbers of nodes in each layer of the first dense network, the "output_dense" is a similary array for the output layer and  the "aggr" is a tuple containing number of aggregation layers (n_aggr) along with the type of method (aggr_type) to be used for aggregation.
            - n_neighbors (int): Number of message passing neighbors for each data point.
            - kernel (str): kernel type, either "gaussian" or "exponential"
        """
        
        self.n_neighbors = n_neighbors

        if kernel not in _allowed_kernels
        self.kernel      = _allowed_kernels[kernel]

