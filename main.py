from __future__ import print_function

from os import mkdir
from multiprocessing import cpu_count

from torch import cuda, device, no_grad, save
from torch.nn import NLLLoss
from torch.optim import SGD

from data import mnist_loader
from nn.SimpleFC import SimpleFC

# select number of cpu workers, maximum of 8
MAX_NB_CPU = 8
NB_CPU = cpu_count() if cpu_count() < MAX_NB_CPU else MAX_NB_CPU

# use cuda if available
use_cuda = cuda.is_available()
device = device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': NB_CPU, 'pin_memory': True} if use_cuda else {}


def train(train_loader, model, optimizer, criterion, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))


def test(test_loader, model, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main():
    # training parameters
    batch_size = 200
    epochs = 10
    learning_rate = 0.01
    log_interval = 10
    momentum = 0.9
    save_model = True

    # create neural network model
    model = SimpleFC(input_shape=[1, 28, 28]).to(device)

    # create a stochastic gradient descent optimizer
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # create a loss function
    criterion = NLLLoss()

    # get training and test data sets
    train_loader = mnist_loader.get_train_loader(batch_size=batch_size, kwargs=kwargs)
    test_loader = mnist_loader.get_test_loader(batch_size=batch_size, kwargs=kwargs)

    # train and test for x epochs
    for epoch in range(1, epochs + 1):
        train(train_loader, model, optimizer, criterion, epoch, log_interval)
        test(test_loader, model, criterion)

    if save_model:
        try:
            mkdir("trained_models")
        except FileExistsError:
            pass

        save(model.state_dict(), "trained_models/mnist_simple_fc.pt")


if __name__ == '__main__':
    main()
