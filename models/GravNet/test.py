import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from torch_geometric.nn import knn_graph

class GravNetMessagePassing(MessagePassing):
    def __init__(self, aggr=['mean', 'max', 'min', 'std']):
        super().__init__(aggr=aggr)

    def forward(self, edge_index: torch.OptionalType) -> torch.OptionalType:
        return self.propagate(edge_index, x=x, w=w)
    
    def message(self, x, w):
        return np.multiply(x, w)
    
x = np.linspace(0,10,11)
w = np.random.rand(11)
x = torch.tensor(x)
#x = torch.Tensor(np.array([x]))
w = torch.tensor(np.array([w]))

edge_index = torch.tensor([[0,1], [0,2], [0,3],
                           [1,0], [1,2], [1,4],
                           [2,0], [2,1], [2,3], [2,4], [2,5],
                           [3,0], [3,2], [3,7],
                           [4,1], [4,2], [4,6], [4,7],
                           [5,2], [5,3], [5,7], [5,8],
                           [6,4], [6,7], [6,9],
                           [7,4], [7,5], [7,6], [7,8], [7,9],
                           [8,5], [8,7], [8,9],
                           [9,6], [9,7], [9,8]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index.t().contiguous())
msg = GravNetMessagePassing()