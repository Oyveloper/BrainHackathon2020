import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

import utils



NET_PATH = './model/model.pth'

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
    # Load data
    data = pd.read_csv("./data/data.2018-10-31.93b9dcde-5e2e-11ea-8d9e-000d3a64d565.csv")
    data.fillna(method='ffill', inplace=True)
    data['turbine'] = data['turbine'].apply(lambda x: utils.turbine_to_number(x))
    division_point = int(len(data) * 0.8)
    train_data = data.iloc[:division_point, :]
    test_data = data.iloc[division_point:, :]

    columns = ['turbine', 'WindSpeed (Average)', 'WindDirection (Average)']

    n_dim = len(columns)


    train_X = train_data[columns] 
    train_Y = train_data['ActivePower (Average)']

    print(train_X.values[0])

    test_X = test_data[columns]
    test_Y = test_data['ActivePower (Average)']


    training_input = torch.tensor(train_X.values)
    training_output = torch.tensor(train_Y.values)



    
    testing_input = torch.tensor(test_X.values)
    testing_output = torch.tensor(test_Y.values)



    input_size = training_input.size()[1]
    hidden_size = 30

    net = Net(input_size, hidden_size)



    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001)

    epochs = 2
    errors = []

    print("Starting training")
    # for epoch in range(epochs):
    #     for i in tqdm(range(training_input.size()[0])):
    #         x = training_input[i]
    #         y = training_output[i]

    #         optimizer.zero_grad()

    #         y_pred = net(x)
    #         if (y_pred != y_pred).any():
    #             print(x)
    #             print(y_pred)
    #             print(net.fc1.weight)
    #             print(y)
    #             return 
            
    #         loss = criterion(y_pred.double(), y.double())

    #         loss.backward()
    #         optimizer.step()
      
        

    print("Training finished")
    torch.save(net.state_dict(), NET_PATH)

    net = Net(input_size, hidden_size)
    net.load_state_dict(torch.load(NET_PATH))

    


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

    
    # Train here




def get_trained_net():
    net = Net()
    net.load_state_dict(torch.load(NET_PATH))

if __name__ == "__main__":
    train_net()

