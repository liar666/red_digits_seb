#!/usr/bin/env python
import pyreadr
#import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time
import datetime

# The unflattened shape of the images in the RData file
DIGIT_SHAPE = { "width": 10, "height": 14, "channels": 3 }

# Columns where the image data is stored in the RData file
IMG_RANGE = np.s_[1:421]
# Columns where the image data is stored in the RData file
CLS_RANGE = np.s_[422:435]

# All the possible outcomes
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "E", "H", "Other"]

# The device where to put the data
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(DEVICE)


class CNN(nn.Module):
    # Create the (Conv) layers & activation functions
    def __init__(self, input_size=(DIGIT_SHAPE["width"], DIGIT_SHAPE["height"]), hidden_size=500, out_size=len(CLASSES)):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=input_size, stride=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=out_size, kernel_size=1, stride=1)
        self.actv1 = nn.ReLU(inplace=True)
        self.actv2 = nn.LogSoftmax(dim=1)

    # Connect the layers
    def forward(self, x):
        x = self.actv1(self.conv1(x))
        x = self.actv2(self.conv2(x))
        return x
        
    def train_me(self, input_data_train, target_train, input_data_test, target_test, optimizer, loss, nb_epochs, batch_size=1):
        data_count = input_data_train.shape[0] 
        assert input_data_train.shape[0] == target_train.shape[0]
    
        start_time = time.time()
            
        for epoch_index in range(nb_epochs):
            # Put the network in train mode
            self.train()
            # Shuffle the data for this epoch
            p = np.random.permutation(data_count)
            # Train by batch
            start_ = 0
            stop_  = 0
            count  = 0
            loss_  = 0
            while stop_ < data_count:
                # Reset the gradients for this batch
                optimizer.zero_grad()
                # Compute the positions for the current batch
                stop_ = min([start_ + batch_size, data_count])
                p_batch = p[start_:stop_]
                start_ =  stop_
                # Extract the data for the current batch
                input_ = input_data_train[p_batch]
                target_ = target_train[p_batch]
                # Comput the CNN output for the batch
                out_ = self.forward(input_)
                out_ = out_.reshape(out_.shape[0], out_.shape[1])
                # Compute the loss for the current batch
                loss_func = loss(out_, target_)
                loss_func.backward()
                optimizer.step()  # Does the update
                loss_ += loss_func.item()
                count += 1
            # Switch the NN to evaluation mode
            self.eval()
            # Evaluate the NN after training for a new epoch
            with torch.no_grad():
                res = self.test_me(input_data_test, target_test, batch_size=1000)
            # Compute a few metrics for the current epoch (time + accuracy)
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
            print ("%d loss %.4f   acc %.2f%%   %s"% (epoch_index + 1, loss_, res["acc"]*100, str_time))

    def test_me(self, input_data, target, batch_size=1):
        data_count = input_data.shape[0]
        assert input_data.shape[0] == target.shape[0]
        start_ = 0
        stop_  = 0
        acc    = 0
        while stop_ < data_count:
            # Compute the positions for the current batch
            stop_ = min([start_ + batch_size, data_count])
            input_ = input_data[start_:stop_]
            target_ = target[start_:stop_]
            start_ =  stop_
            # Compute the output of the CNN on the current batch
            out_ = self.forward(input_)
            out_ = out_.reshape(out_.shape[0], out_.shape[1])
            out_class = torch.argmax(out_, dim=1)
            # Compute the number of correct responses for this batch
            acc += torch.sum(out_class ==  target_).item()
        # Compute the overall accurracy
        acc /= data_count
        res = {'acc':acc}
        return res


def get_input_target(data, device):
    input_data_list = []
    target_list = []

    # Reshape image values as image of size number_height*number_width with 3 channels & Compute mean of 3 channels
    input_data_list = [torch.sum(torch.tensor(sample[IMG_RANGE].reshape(DIGIT_SHAPE["channels"], DIGIT_SHAPE["height"], DIGIT_SHAPE["width"]),dtype=torch.float), dim=0, keepdim=True)/3 for sample in data_train]
    # Compute Class value based on Class 1-hot encoding
    target_list = [np.argmax(sample[CLS_RANGE]).item() for sample in data_train]
    
    # Put data on the GPU
    input_data = torch.stack(input_data_list).to(device)
    target = torch.tensor(target_list).to(device)

    return input_data, target


if __name__ == "__main__":
    # Read data from the R backup
    print(">> Reading data from the RData file")
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
    
    # Formatting data for the CNN
    print(">> Formatting data for the CNN")
    input_data_train, target_train = get_input_target(data_train, DEVICE)
    input_data_test, target_test = get_input_target(data_test, DEVICE)
    
    
    # A few parameters for the CNN
    mu          = 0.001
    input_size  = (DIGIT_SHAPE["height"], DIGIT_SHAPE["width"]) # 140
    hidden_size = 500
    out_size    = len(CLASSES)
    batch_size  = 1000
    nb_epochs   = 100

    # Create CNN and put it on GPU 
    cnn = CNN(input_size, hidden_size, out_size).to(DEVICE)
    # Define the Loss as the CrossEntropy
    nn_loss = torch.nn.CrossEntropyLoss()
    nn_optimizer = torch.optim.Adam(
      list(cnn.parameters()), lr=mu
    )
    
    print(">> Training the CNN")
    cnn.train_me(input_data_train=input_data_train, target_train=target_train,
                 input_data_test=input_data_test, target_test=target_test,
                 optimizer=nn_optimizer, loss=nn_loss,
                 nb_epochs=nb_epochs, batch_size=batch_size)
