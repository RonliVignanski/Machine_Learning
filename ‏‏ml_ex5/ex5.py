# 211545892 Ronli Vignanski
# 316383298 Yarin Dado

import sys
import os
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from gcommand_dataset import GCommandLoader
import matplotlib.pyplot as plt


# our implementation of VGG11 architecture according to the specification
class MyVGG11(nn.Module):
    """
      Create a model of VGG11 architecture, in order to train the data set with it.
      It is with 11 layers, including cnn and pooling, so that after each layer of
      convolution we do the ReLU activation in order to make the network not linear.
      """
    def __init__(self, in_channels, num_classes=1000):
        super(MyVGG11, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=7680, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # flatten to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return F.log_softmax(x, dim=1)


def train(loaded_data, model, optimizer, cuda):
    """
    training the model.
    Args:
        loaded_data: train data loader
        model: VGG11
        optimizer: ADAM
        epoch:
        cuda: boolean that shows if we run on cuda or not
    Returns: None
    """
    model.train()
    for _, (data, target) in enumerate(loaded_data):
        criterion = nn.CrossEntropyLoss()
        if cuda:
            data, target = data.cuda(), target.cuda()
            criterion = criterion.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(loaded_data, model, cuda):
    """
    test the new data according to the trained data.
    Args:
        loaded_data: train data loader
        model: VGG11
        cuda:  boolean that shows if we run on cuda or not

    Returns:

    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaded_data:
            criterion = nn.CrossEntropyLoss(reduction='sum')
            if cuda:
                data, target = data.cuda(), target.cuda()
                criterion = criterion.cuda()
            output = model(data)
            test_loss += criterion(output, target).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loaded_data.dataset)
    accuracy = float(correct) / len(loaded_data.dataset)

    return test_loss, accuracy


def predict(model, test_x, file_path, classes, spects):
    """
    predict the labels for the test_x data, and write it to test_y file.
    :param model: model we trained
    :param test_x: data we want to predict
    :param file_path: output file
    :return: None
    """
    with open(file_path, "w") as file:
        model.eval()
        with torch.no_grad():
            for data, spect in zip(test_x, spects):
                if cuda:
                    data = data.cuda()
                output = model(data)
                prediction = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                prediction_index = prediction.view(-1).item()
                prediction_name = classes[prediction_index]
                file_name = os.path.basename(spect)
                # writing the name file with the prediction
                file.write(f'{file_name},{prediction_name}\n')


if __name__ == '__main__':
    submission_mode = True

    # EXAMPLE: python3 ex5.py ./data/train ./data/valid ./data/test ./test_y
    train_path, valid_path, test_path, out_file = sys.argv[1:5]

    cuda = torch.cuda.is_available()

    # load data
    train_dataset = GCommandLoader(train_path)
    # list of class names
    classes = train_dataset.classes
    valid_dataset = GCommandLoader(valid_path)
    test_dataset = GCommandLoader(test_path, loadtest=True)
    # list of input file paths
    spects = test_dataset.spects

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4,
                                               pin_memory=True, sampler=None)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4,
                                               pin_memory=True, sampler=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                                              pin_memory=True, sampler=None)

    model = MyVGG11(1, 30)
    if cuda:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = opt.Adam(model.parameters(), lr=0.0003)

    # train the model
    epochs = 40
    loss_data = []
    accuracy_data = []
    for e in range(epochs):
        train(train_loader, model, optimizer, cuda)
        if not submission_mode:
            test_loss, accuracy = test(valid_loader, model, cuda)
            loss_data.append(test_loss)
            accuracy_data.append(accuracy)

    if not submission_mode:
        # create plots
        plt.plot(list(range(1, epochs + 1)), accuracy_data, color='red', label='accuracy')
        plt.title('Accuracy Per Epoch')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(f'./accuracy.png')
        plt.clf()
        plt.plot(list(range(1, epochs + 1)), loss_data, color='red', label='loss')
        plt.title('Loss Per Epoch')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(f'./loss.png')

    # output predictions on the test data to an external file
    predict(model, test_loader, out_file, classes, spects)

