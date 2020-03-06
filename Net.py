import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score



NET_PATH = './model/model.pth'

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
        x = self.sigmoid(self.fc3(x))
        print(x)
        return x



def train_net():
    """
    Trains the neural network on the best data
    """
    # Load data
    data = pd.read_csv("./data/data.2018-10-31.93b9dcde-5e2e-11ea-8d9e-000d3a64d565.csv")
    train_data = data.iloc[:102290, :]
    test_data = data.iloc[102290:, :]


    train_X = train_data['WindSpeed (Average)'] 
    train_Y = train_data['ActivePower (Average)']

    test_X = test_data['WindSpeed (Average)']
    test_Y = test_data['ActivePower (Average)']


    training_input = torch.tensor(train_X.values).view(-1, 1)
    training_output = torch.tensor(train_Y.values).view(-1, 1)

    testing_input = torch.tensor(test_X.values).view(-1, 1)
    testing_output = torch.tensor(test_Y.values).view(-1, 1)

    
    # training_input = torch.rand(100, 3)
    # training_output = torch.rand(100, 3)

    # raining_input = training_input.view(-1)
    # training_output = training_output.view(-1)




    input_size = training_input.size()[1]
    hidden_size = 1000

    net = Net(input_size, hidden_size)



    criterion = torch.nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.0000001)

    epochs = 2
    errors = []
    for epoch in range(epochs):
        for i in tqdm(range(training_input.size()[0])):
            x = training_input[i]
            y = training_output[i]

            optimizer.zero_grad()

            y_pred = net(x)
            loss = criterion(y_pred, y)
            print(loss)
            loss.backward()
            optimizer.step()
      
        

    print("Training finished")
    torch.save(net.state_dict(), NET_PATH)

    net = Net(input_size, hidden_size)
    net.load_state_dict(torch.load(NET_PATH))
    print(net(training_input[0]))
    

    total = 0
    correct = 0
    with torch.no_grad():
        for i in tqdm(range(testing_input.size()[0])):
            x = testing_input[i]
            y = testing_output[i]
            output = net(x)
            total += 1

            correct += (output == y)

    print('Accuracy of the network: %d %%' % (
    100 * correct / total))
    
    # Train here




def get_trained_net():
    net = Net()
    net.load_state_dict(torch.load(NET_PATH))

if __name__ == "__main__":
    train_net()

