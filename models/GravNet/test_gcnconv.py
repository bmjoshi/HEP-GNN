import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.data import Data

class GravNetMessagePassing(MessagePassing):
    def _init_(self):
        super().__init__(aggr='add')
    def forward(self, x, edge_index):
    def message(self, x, w):
        return torch.multiply(x, w)

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr=['add','max'])  # "Add" aggregation (Step 5).
        self.aggr_channel_size = out_channels*len(self.aggr)
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(self.aggr_channel_size))
        self.dense_out = Linear(self.aggr_channel_size, 3)
        self.reset_parameters()
        print('Done Reset')

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        print(x)
        print(edge_index.shape)
        new_var = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_index, _ = new_var
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        print(edge_index)
        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        print(norm)
        print('x before: ', x)
        print('edge_index:', edge_index.shape)
        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        print('out: ', out)
        # Step 6: Apply a final bias vector.
        print(self.bias)
        out += self.bias

        out = self.dense_out(out)        

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        print('x_j.shape: ', x_j.shape)
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
#x = torch.tensor([-1,0,1], dtype=torch.float)
x = torch.rand([3, 3])
# data = Data(x=x, edge_index=edge_index.t().contiguous())
conv = GCNConv(3, 16)
res = conv(x, edge_index=edge_index.t().contiguous())
print(res)
