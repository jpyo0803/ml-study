from torch_geometric.datasets import Planetoid

def load_dataset():
    return Planetoid(root='data/processed', name='Cora')
