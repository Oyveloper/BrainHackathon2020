import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from FetchData import FetchData
import utils



NET_PATH = './model/model.pth'

dataFetcher = FetchData()
hidden_size = 40

class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.fc3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train_net():
    """
    Trains the neural network on the best data
    """

    training_input, training_output = dataFetcher.load_training_data()


   
    input_size = training_input.size()[1]
    net = Net(input_size, hidden_size)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.003)

    epochs = 25
    errors = []

    print("Starting training")
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1}")
        for i in tqdm(range(training_input.size()[0])):
            x = training_input[i]
            y = training_output[i]

            optimizer.zero_grad()

            y_pred = net(x)
            if (y_pred != y_pred).any():
                print(x)
                print(y_pred)
                print(net.fc1.weight)
                print(y)
                return 
            
            loss = criterion(y_pred.double(), y.double())

            loss.backward()
            optimizer.step()
      
        

    print("Training finished")
    torch.save(net.state_dict(), NET_PATH)

   
    

def test_net():
    # get the data
    testing_input, testing_output = dataFetcher.load_testing_data()



    net = get_trained_net()

    print("Starting to test the model")
    total = 0
    differnces = []
    with torch.no_grad():
        for i in tqdm(range(testing_input.size()[0])):
            x = testing_input[i]
            y = testing_output[i]
            output = net(x)
            y_val = y.item()
            output_val = output.item()
            difference = abs((output_val-y_val))
            differnces.append(difference)
            total += 1


    average_difference = sum(differnces) / len(differnces)
    print(f"Average difference to correct answer was: {average_difference}")
    
    


def get_trained_net():
    input_size = dataFetcher.get_inputsize()
    net = Net(input_size, hidden_size)
    net.load_state_dict(torch.load(NET_PATH))
    return net


