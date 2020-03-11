import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from FetchData import FetchData
from HistoryDataset import HistoryDataset
import utils

import math



NET_PATH = './model/model.pth'

dataFetcher = FetchData()
hidden_size = 18
epochs = 10
lr = 0.00001
batch_size= 20



class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = self.fc3(x)
        return x





def train_net():
    """
    Trains the neural network on the best data
    """

    training_dataset = HistoryDataset()
    training_loader = DataLoader(training_dataset, batch_size=batch_size, num_workers=2)

    # training_input, training_output = dataFetcher.load_training_data()

    input_size = training_dataset.get_element_size()
    net = Net(input_size, hidden_size)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr)



    # allowing for gpu training
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net.to(device)
    # print(device)


    errors = []

    print("Starting training")
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1}")

        with tqdm(total=math.ceil(len(training_dataset)/batch_size)) as pbar:
            for i, data in enumerate(training_loader):
                x = data[0] #training_input[i]#.to(device)
                y = data[1] #training_output[i]#.to(device)

                optimizer.zero_grad()

                y_pred = net(x)
                if (y_pred != y_pred).any():
                    print(x)
                    print(y_pred)
                    print(y)
                    return 

                loss = criterion(y_pred.double(), y.double())

                loss.backward()
                optimizer.step()
                pbar.update(1)
      
        

    print("Training finished")

    torch.save(net.state_dict(), NET_PATH)




import math

def test_net():

    test_dataset = HistoryDataset(training=False)
    test_loader = DataLoader(test_dataset, num_workers=2, batch_size=batch_size)
    

    net = get_trained_net()

    print("Starting to test the model")
    total = 0
    differences = []
    with torch.no_grad():
        with tqdm(total=math.ceil(len(test_dataset) / batch_size)) as pbar:
            for i, data in enumerate(test_loader):
                x = data[0]
                y = data[1]
                output = net(x)
                vals = [v.item() for v in y]
                output_vals = [v.item() for v in output]

                differences += [a - b for a, b in zip(vals, output_vals)]
                total += 5
                pbar.update(1)
            


    

    average_difference = sum(differences) / len(differences)
    print(f"Average difference to correct answer was: {average_difference}")
    





def get_trained_net():
    input_size = dataFetcher.get_inputsize()
    net = Net(input_size, hidden_size)
    net.load_state_dict(torch.load(NET_PATH))
    return net


