
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.params = {
            'optimizer': 'sgd',
            'learning_rate': 0.000001,
            'epochs': 10,
            'batch_size': 128,
            'dropouts': []
        }
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.params = {
            'optimizer': 'adam',
            'learning_rate': 0.0001,
            'epochs': 10,
            'batch_size': 128,
            'dropouts': []
        }
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


class ModelC(nn.Module):
    def __init__(self):
        super(ModelC, self).__init__()
        self.params = {
            'optimizer': 'adam',
            'learning_rate': 0.0001,
            'epochs': 10,
            'batch_size': 32,
            'dropouts': [0.1, 0.1]
        }
        self.dropout1 = nn.Dropout(self.params['dropouts'][0])
        self.dropout2 = nn.Dropout(self.params['dropouts'][1])
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.params = {
            'optimizer': 'adam',
            'learning_rate': 0.0001,
            'epochs': 10,
            'batch_size': 128,
            'dropouts': []
        }
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


class ModelE(nn.Module):
    def __init__(self):
        super(ModelE, self).__init__()
        self.params = {
            'optimizer': 'adam',
            'learning_rate': 0.0001,
            'epochs': 10,
            'batch_size': 32,
            'dropouts': []
        }
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        output = F.log_softmax(x, dim=1)
        return output


class ModelF(nn.Module):
    def __init__(self):
        super(ModelF, self).__init__()
        self.params = {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 32,
            'dropouts': []
        }
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        x = self.fc5(x)
        x = torch.sigmoid(x)
        x = self.fc6(x)
        output = F.log_softmax(x, dim=1)
        return output


class ModelBest_model(nn.Module):
    """
    Model that we created in order to have best results.
    Using validation we succeeded to find best parameters including number of epochs, probabilities in dropouts,
    learning rate, batch size, optimizer, number of layers, layer sizes.
    """
    def __init__(self):
        super(ModelBest_model, self).__init__()
        self.params = {
            'optimizer': 'adam',
            'learning_rate': 0.0003,
            'epochs': 50,
            'batch_size': 128,
            'dropouts': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        }
        self.dropout1 = nn.Dropout(self.params['dropouts'][0])
        self.dropout2 = nn.Dropout(self.params['dropouts'][1])
        self.dropout3 = nn.Dropout(self.params['dropouts'][2])
        self.dropout4 = nn.Dropout(self.params['dropouts'][3])
        self.dropout5 = nn.Dropout(self.params['dropouts'][4])
        self.dropout6 = nn.Dropout(self.params['dropouts'][5])

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(16)

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader):
    """

    :param model: model we need to train
    :param device: cpu
    :param train_loader: data loader for the train data
    :return: avg train loss and avg train accuracy
    """
    optimizer = get_optimizer(model)
    train_loader_size = len(train_loader.dataset)

    model.train()
    train_correct = 0
    train_loss_sum = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        train_loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        prediction = output.argmax(dim=1, keepdim=True)
        train_correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss_sum / train_loader_size, train_correct / train_loader_size


def test(model, device, test_loader):
    """

    :param model: model we want to test
    :param device: cpu
    :param test_loader: data loader for the test data
    :return: avg test loss and avg test accuracy
    """
    model.eval()

    val_correct = 0
    val_loss_sum = 0
    val_loader_size = len(test_loader.dataset)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            val_loss_sum += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            prediction = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            val_correct += prediction.eq(target.view_as(prediction)).sum().item()

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        val_loss_sum, val_correct, len(test_loader.dataset),
        100. * val_correct / len(test_loader.dataset)))
    return val_loss_sum / val_loader_size, val_correct / val_loader_size, 100. * val_correct / len(test_loader.dataset)


def write_log(file_name, lr, batch, optimizer, dropout, accuracy, epoch):
    """
    documents the hyper-parameters of each run. writes to a local file
    :return:
    """
    log_file = open(file_name, "a")
    output = f"----------------------------\n"
    output += f"PARAMETERS:\n"
    output += f"NUM EPOCHS = " + str(epoch) + "\n"
    output += f"RATE = " + str(lr) + "\n"
    output += f"BATCH_SIZE = " + str(batch) + "\n"
    output += f"OPTIMIZER = " + str(optimizer) + "\n"
    if file_name == "logQ":
        output += f"DROPOUT = " + str(dropout[0]) + "," + str(dropout[1]) + str(dropout[2]) + str(dropout[3]) + str(
            dropout[4]) + str(dropout[5])
        "\n"
    if file_name == 'C':
        output += f"DROPOUT = " + str(dropout[0]) + "," + str(dropout[1]) + "\n"
    output += f" !! RESULT ALL(%): " + str(accuracy) + "\n"

    log_file.write(output)
    log_file.close()


def get_optimizer(model):
    """
    get the optimizer.
    :param model: the model we check
    :return: the optimizer
    """
    optimizer_type = model.params['optimizer']
    learning_rate = model.params['learning_rate']
    if optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))


def plot_graph_accuracy(model, name, train_data, val_data):
    """
    show a graph for the train and validation accuracy.
    :param model: model we check
    :param name: name of the model
    :param train_data: train data
    :param val_data: validation data
    :return: None
    """
    epochs = [e for e in range(1, model.params['epochs'] + 1)]
    label1 = 'train avg. accuracy'
    label2 = 'validation avg. accuracy'
    title = f'Model {name}: Avg. Accuracy per Epoch'
    plt.figure(1)
    plt.plot(epochs, train_data, color='red', label=label1)
    plt.plot(epochs, val_data, color='blue', label=label2)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('average accuracy')
    plt.legend()
    plt.savefig(f'./graphs/Model_{name}_accuracy_real.png')
    # plt.show()
    plt.close()


def plot_graph_loss(model, name, train_data, val_data):
    """
       show a graph for the train and validation loss.
       :param model: model we check
       :param name: name of the model
       :param train_data: train data
       :param val_data: validation data
       :return: None
       """
    epochs = [e for e in range(1, model.params['epochs'] + 1)]
    label1 = 'train avg. loss'
    label2 = 'validation avg. loss'
    title = f'Model {name}: Avg. Loss per Epoch'
    plt.figure(2)
    plt.plot(epochs, train_data, color='red', label=label1)
    plt.plot(epochs, val_data, color='blue', label=label2)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('average loss')
    plt.legend()
    plt.savefig(f'./graphs/Model_{name}_loss_real.png')
    # plt.show()
    plt.close()


def predict(model, test_x, file_path):
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
            for data in test_x[:-1]:
                output = model(data)
                prediction = output.argmax(dim=1, keepdim=True)
                file.write(f'{prediction.view(-1).item()}\n')
            output = model(test_x[-1])
            prediction = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            file.write(f'{prediction.view(-1).item()}')


def main():

    device = torch.device("cpu")
    # normalize the data
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # loading the data and normalize it
    train_x_path, train_y_path, test_x_path, output_file = sys.argv[1:5]
    train_x = trans(np.loadtxt(train_x_path)).float()
    train_x = torch.squeeze(train_x)
    train_y = torch.from_numpy(np.loadtxt(train_y_path)).long()
    test_x = trans(np.loadtxt(test_x_path)).float()
    test_x = torch.squeeze(test_x)

    models = {
        'A': ModelA().to(device),
        'B': ModelB().to(device),
        'C': ModelC().to(device),
        'D': ModelD().to(device),
        'E': ModelE().to(device),
        'F': ModelF().to(device),
        'Best_model': ModelBest_model().to(device)
    }

    show_graphs = False

    if show_graphs:

        # for validation
        dataset = TensorDataset(train_x, train_y)
        data_size = len(dataset)
        validation_size = 0.2
        train_size = int((1 - validation_size) * data_size)
        test_size = data_size - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        # for checking
        for key in models.keys():
            print(key)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=models[key].params['batch_size'],
                                                       shuffle=True)

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=models[key].params['batch_size'],
                                                      shuffle=False)

            list_train_accuracy = []
            list_val_accuracy = []
            list_train_loss = []
            list_val_loss = []
            for epoch in range((models[key]).params['epochs']):
                print("IN EPOCH ", epoch)
                train_loss, train_accuracy = train(models[key], device, train_loader)
                list_train_loss.append(train_loss)
                list_train_accuracy.append(train_accuracy)
                val_loss, val_accuracy, accuracy = test(models[key], device, test_loader)
                list_val_loss.append(val_loss)
                list_val_accuracy.append(val_accuracy)
                write_log("log" + key, models[key].params['learning_rate'], models[key].params['batch_size'],
                          models[key].params['optimizer'],
                          models[key].params['dropouts'], accuracy, models[key].params['epochs'])
            # create the plots for each model
            plot_graph_accuracy(models[key], key, list_train_accuracy, list_val_accuracy)
            plot_graph_loss(models[key], key, list_train_loss, list_val_loss)
    else:
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=models['Best_model'].params['batch_size'],
                                                   shuffle=True)
        for _ in range(models['Best_model'].params['epochs']):
            train(models['Best_model'], device, train_loader)

        predict(models['Best_model'], test_x, output_file)


if __name__ == '__main__':
    main()
