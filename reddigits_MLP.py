import pyreadr
#import pandas as pd
import numpy as np
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
import time
import datetime

# The unflattened shape of the images in the RData file
NUMBER_SHAPE = { "width": 10, "height": 14, "channels": 3 }

# Columns where the image data is stored in the RData file
IMG_RANGE = np.s_[1:421]
# Columns where the image data is stored in the RData file
CLS_RANGE = np.s_[422:435]

CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "E", "H", "Other"]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(DEVICE)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size[0]*input_size[1], hidden_size)
        self.hdd_actv = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, out_size)
        self.out_actv = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Flatten image input
        x = x.view(-1, NUMBER_SHAPE["height"] * NUMBER_SHAPE["width"])
        # Connect the layers
        x = self.hdd_actv(self.fc1(x))
        x = self.out_actv(self.fc2(x))
        return x

    def train_me(self, input_data_train, target_train, input_data_test, target_test, nn_optimizer, criterion, batch_size=1):
        data_count = input_data_train.shape[0] 
        assert input_data_train.shape[0] == target_train.shape[0]
    
        start_time = time.time()
            
        for epoch_index in range(nb_epochs):
            # Put the MLP in train mode
            self.train()
            # Shuffle the input data
            p = np.random.permutation(data_count)
            
            start_ = 0
            stop_  = 0
            count  = 0
            loss_  = 0
            while stop_ < data_count:
                # Reset the gradients between each batch
                nn_optimizer.zero_grad()
                
                # Compute the positions of the current batch
                stop_ = min([start_ + batch_size, data_count])
                p_batch = p[start_:stop_]
                start_ =  stop_
                
                # Extract the current batch info
                input_ = input_data_train[p_batch]
                target_ = target_train[p_batch]
    
                # Run the forward pass on the MLP with the batch data
                out_ = self.forward(input_)
    
                # Compute the loss on the batch data
                loss = criterion(out_, target_)
                loss.backward()
                nn_optimizer.step()  # Does the update
                loss_ += loss.item()
    
                count += 1
            
            # Switch the MLP into evaluation mode
            self.eval()
            
            # Test the NLP performance after this epoch
            with torch.no_grad():
                res = self.test_me(input_data_test, target_test, batch_size=1000)
    
            # Display MLP performance metrics (time + accuracy) after this epoch
            elapsed_time = time.time() - start_time
            expected_time = elapsed_time * nb_epochs / (epoch_index + 1)
            loss_ /= count
            str_time = (
                "( "
                + str(datetime.timedelta(seconds=round(elapsed_time)))
                + " / "
                + str(datetime.timedelta(seconds=round(expected_time)))
                + " )"
            )
            print ("%d loss %.4f   rec %.2f%%   %s"% (epoch_index + 1, loss_, res["acc"]*100, str_time))

    def test_me(self, input_data, target, batch_size=1):
        data_count = input_data.shape[0]
        assert input_data.shape[0] == target.shape[0]
        start_ = 0
        stop_  = 0
        acc    = 0
        while stop_ < data_count:
            # Compute batch positions
            stop_ = min([start_ + batch_size, data_count])
            input_ = input_data[start_:stop_]
            target_ = target[start_:stop_]
            start_ =  stop_
            # Compute MLP output on current batch
            out_class = torch.argmax(self.forward(input_), dim=1)
            # Compute number of correct outputs in this batch
            acc += torch.sum(out_class ==  target_).item()
    
        # Compute overall Accurracy
        acc /= data_count
        acc = {'acc': acc}
        return acc



def get_input_target(data, device):
    input_data_list = []
    target_list = []

    # Reshape image values as image of size digit_height*digit_width with 3 channels & Compute mean of 3 channels
    input_data_list = [torch.sum(torch.tensor(sample[IMG_RANGE].reshape(NUMBER_SHAPE["channels"], NUMBER_SHAPE["height"], NUMBER_SHAPE["width"]),dtype=torch.float), dim=0, keepdim=True)/3 for sample in data_train]
    # Compute Class value based on Class 1-hot encoding
    target_list = [np.argmax(sample[CLS_RANGE]).item() for sample in data_train]
    
    # Put data on the GPU
    input_data = torch.stack(input_data_list).to(device)
    target = torch.tensor(target_list).to(device)

    return input_data, target


if __name__ == "__main__":
    # Read data from the R backup
    print(">> Loading RData file")
    data = pyreadr.read_r('./Data/RData/train+testWhole.RData') # also works for Rds
    data_train = data['train'].values
    data_test  = data['test'].values
    
    # Code to plot image to try to find actual shape of images
    # plt.figure()
    # img = np.array(data_test[10][1:421]).reshape(3,14,10).sum(axis=0)/3
    # plt.xticks([])
    # plt.yticks([])
    # plt.grid(False)
    # plt.imshow(img)
    # plt.show()
    
    print(">> Formatting data for MLP")
    input_data_train, target_train = get_input_target(data_train, DEVICE)
    input_data_test, target_test = get_input_target(data_test, DEVICE)
    
    # Define a few parameters for the MLP
    mu          = 0.001
    input_size  = (NUMBER_SHAPE["height"], NUMBER_SHAPE["width"]) # 140
    hidden_size = 500
    out_size    = len(CLASSES)
    batch_size  = 1000
    nb_epochs   = 100

    # Create MLP and put it on GPU
    print(">> Creating MLP")
    mlp = MLP(input_size, hidden_size, out_size).to(DEVICE)
    # Define the Loss as the CrossEntropy
    nn_loss = torch.nn.CrossEntropyLoss()
    nn_optimizer = torch.optim.Adam(
        list(mlp.parameters()), lr=mu
        )
    # Train the MLP
    print(">> Training MLP")
    mlp.train_me(input_data_train=input_data_train, target_train=target_train,
              input_data_test=input_data_test,   target_test=target_test,
              nn_optimizer=nn_optimizer, criterion=nn_loss, batch_size=batch_size)
    