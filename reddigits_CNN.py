import pyreadr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import datetime


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


def get_input_target(data, device):
    input_data_list = []
    target_list = []

    input_data_list = [torch.sum(torch.tensor(sample[1:421].reshape(3, 14,10),dtype=torch.float), dim=0, keepdim=True)/3 for sample in data_train]
    target_list = [np.argmax(sample[422:435]).item() for sample in data_train]

    input_data = torch.stack(input_data_list).to(device)
    target = torch.tensor(target_list).to(device)

    return input_data, target


data = pyreadr.read_r('./data/train+testWhole.RData') # also works for Rds

# plt.figure()

data_train = data['train'].values
data_test = data['test'].values

# img_ = np.array(data_test[10][1:421]).reshape(3,14,10).sum(axis=0)/3

input_data_train, target_train = get_input_target(data_train, device)
input_data_test, target_test = get_input_target(data_test, device)


# plt.xticks([])
# plt.yticks([])
# plt.grid(False)
# plt.imshow(img_)
# plt.show()

mu=0.001
input_size = (14, 10) # 140
hidden_size = 500
out_size = 13
batch_size = 1000
epoch_count = 100

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.actv = nn.ReLU(True)
        self.fc2 = nn.Linear(hidden_size, out_size)
        self.out_actv = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.actv(self.fc1(x))
        x = self.fc2(x)
        x = self.out_actv(x)
        return x


# mlp = MLP(input_size, hidden_size, out_size).to(device)

# criterion = torch.nn.CrossEntropyLoss()
# nn_optimizer = torch.optim.Adam(
#   list(mlp.parameters()), lr=mu
# )

class CNN(nn.Module):
    def __init__(self, input_size=(10, 14), hidden_size=500, out_size=13):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_size, input_size, 1)
        self.conv2 = nn.Conv2d(hidden_size, out_size, 1, 1)
        self.actv1 = nn.ReLU(True)
        self.actv2 = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.actv1(self.conv1(x))
        x = self.actv2(self.conv2(x))
        return x

cnn = CNN(input_size, hidden_size, out_size).to(device)

criterion = torch.nn.CrossEntropyLoss()
nn_optimizer = torch.optim.Adam(
  list(cnn.parameters()), lr=mu
)

def test_cnn(input_data, target, cnn, batch_size=1):
    data_count = input_data.shape[0]
    assert input_data.shape[0] == target.shape[0]
    start_ = 0
    stop_ = 0
    rec = 0
    while stop_ < data_count:
        stop_ = min([start_ + batch_size, data_count])
        input_ = input_data[start_:stop_]
        target_ = target[start_:stop_]
        start_ =  stop_
        out_ = cnn.forward(input_)
        out_ = out_.reshape(out_.shape[0], out_.shape[1])
        out_class = torch.argmax(out_, dim=1)
        rec += torch.sum(out_class ==  target_).item()
    
    rec /= data_count
    res = {'rec':rec}

    return res

def test_mlp(input_data, target, mlp, batch_size=1):
    data_count = input_data.shape[0]
    assert input_data.shape[0] == target.shape[0]
    start_ = 0
    stop_ = 0
    rec = 0
    while stop_ < data_count:
        stop_ = min([start_ + batch_size, data_count])
        input_ = input_data[start_:stop_]
        target_ = target[start_:stop_]
        start_ =  stop_
        out_class = torch.argmax(mlp.forward(input_), dim=1)
        rec += torch.sum(out_class ==  target_).item()
    
    rec /= data_count
    res = {'rec':rec}

    return res



def train_mlp(input_data_train, target_train, input_data_test, target_test, mlp, nn_optimizer, criterion, batch_size=1):
    data_count = input_data_train.shape[0] 
    assert input_data_train.shape[0] == target_train.shape[0]

    start_time = time.time()
        
    for epoch_index in range(epoch_count):
        mlp.train()

        p = np.random.permutation(data_count)
        start_ = 0
        stop_ = 0
        count = 0
        loss_ = 0
        while stop_ < data_count:
            nn_optimizer.zero_grad()
            stop_ = min([start_ + batch_size, data_count])
            p_batch = p[start_:stop_]
            start_ =  stop_
            input_ = input_data_train[p_batch]
            target_ = target_train[p_batch]

            out_ = mlp.forward(input_)

            loss = criterion(out_, target_)
            count += 1
            loss.backward()
            nn_optimizer.step()  # Does the update
        
            loss_ += loss.item()

        mlp.eval()
        with torch.no_grad():
            res = test_mlp(input_data_test, target_test, mlp, batch_size=1000)

        elapsed_time = time.time() - start_time
        expected_time = elapsed_time * epoch_count / (epoch_index + 1)

        loss_ /= count

        str_time = (
            "( "
            + str(datetime.timedelta(seconds=round(elapsed_time)))
            + " / "
            + str(datetime.timedelta(seconds=round(expected_time)))
            + " )"
        )


        print ("%d loss %.4f   rec %.2f%%   %s"% (epoch_index + 1, loss_, res["rec"]*100, str_time))


def train_cnn(input_data_train, target_train, input_data_test, target_test, cnn, nn_optimizer, criterion, batch_size=1):
    data_count = input_data_train.shape[0] 
    assert input_data_train.shape[0] == target_train.shape[0]

    start_time = time.time()
        
    for epoch_index in range(epoch_count):
        cnn.train()

        p = np.random.permutation(data_count)
        start_ = 0
        stop_ = 0
        count = 0
        loss_ = 0
        while stop_ < data_count:
            nn_optimizer.zero_grad()
            stop_ = min([start_ + batch_size, data_count])
            p_batch = p[start_:stop_]
            start_ =  stop_
            input_ = input_data_train[p_batch]
            target_ = target_train[p_batch]

            out_ = cnn.forward(input_)
            out_ = out_.reshape(out_.shape[0], out_.shape[1])

            loss = criterion(out_, target_)
            count += 1
            loss.backward()
            nn_optimizer.step()  # Does the update
        
            loss_ += loss.item()

        cnn.eval()
        with torch.no_grad():
            res = test_cnn(input_data_test, target_test, cnn, batch_size=1000)

        elapsed_time = time.time() - start_time
        expected_time = elapsed_time * epoch_count / (epoch_index + 1)

        loss_ /= count

        str_time = (
            "( "
            + str(datetime.timedelta(seconds=round(elapsed_time)))
            + " / "
            + str(datetime.timedelta(seconds=round(expected_time)))
            + " )"
        )


        print ("%d loss %.4f   rec %.2f%%   %s"% (epoch_index + 1, loss_, res["rec"]*100, str_time))


#train_mlp(input_data_train, target_train, input_data_test, target_test, mlp, nn_optimizer=nn_optimizer, criterion=criterion, batch_size=batch_size)

train_cnn(input_data_train, target_train, input_data_test, target_test, cnn, nn_optimizer=nn_optimizer, criterion=criterion, batch_size=batch_size)


            
def read_images(dbd_filename):
    images_list =[]
    return images_list

def images_pyramide(image, ratio):
    return 







