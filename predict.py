from Net import get_trained_net
import torch
from utils import *

def predict_48(from_time):
    pass


net = get_trained_net()
input = torch.tensor([1, 100, 1000])

predict = net(input)
