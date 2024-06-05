import torch
import numpy as np

from models.GravNet.NetworkMap import GravNetMap
from models.GravNet.GravNet import GravNet

x = torch.rand(4, 11)

test_map = GravNetMap()
test_map.setNetworkMap('models/GravNet/config/gravNet_test_config.json')
d = test_map.getNetworkMap()

n = GravNet(test_map)
n(x)

# edge_index = torch.tensor([[0,1], [0,2], [0,3],
#                            [1,0], [1,2], [1,4],
#                            [2,0], [2,1], [2,3], [2,4], [2,5],
#                            [3,0], [3,2], [3,7],
#                            [4,1], [4,2], [4,6], [4,7],
#                            [5,2], [5,3], [5,7], [5,8],
#                            [6,4], [6,7], [6,9],
#                            [7,4], [7,5], [7,6], [7,8], [7,9],
#                            [8,5], [8,7], [8,9],
#                            [9,6], [9,7], [9,8]], dtype=torch.long)
#data = Data(x=x, edge_index=edge_index.t().contiguous())
#msg = GravNetMessagePassing()