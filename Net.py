import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



NET_PATH = './model/model.pth'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        return x


def train_net():
    """
    Trains the neural network on the best data
    """
    net = Net()

    # Train here
    torch.save(net.state_dict(), NET_PATH)

    pass

def get_trained_net():
    net = Net()
    net.load_state_dict(torch.load(NET_PATH))
