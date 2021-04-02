from __future__ import print_function
from os import mkdir
from torchvision import datasets, transforms
import torch as t
from models.Model1 import Model1
from models.Model2 import Model2


kwargs = {'num_workers': 1, 'pin_memory': True}

def train(traingsdata, model, epoch):
    model.train() #weights changing over time due to training
    for batch_idx, (data, target) in enumerate(traingsdata):
        optimizer = t.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        out = model(data)
        criterion = t.nn.NLLLoss()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(traingsdata.dataset), 100. * batch_idx / len(traingsdata),
            loss.item()))


def test(testdata, model):
    model.eval()  #fixed weights for evaluation
    test_loss = 0
    correct = 0
    with t.no_grad():
        for data, target in testdata:
            output = model(data)
            criterion = t.nn.NLLLoss()
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testdata.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testdata.dataset), 100. * correct / len(testdata.dataset)))


def main():

    #(un)comment preferred model
    # sigmoid activation function
    model = Model1(input_shape=[])

    #relu activation function
    #model = Model2(input_shape=[])


    # Aus Kursfolien übernommen
    trainingsdata = t.utils.data.DataLoader(datasets.MNIST('mnist_data_sets', train=True, download=True,
                    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
                    batch_size=64, shuffle=True, **kwargs)

    testdata = t.utils.data.DataLoader(datasets.MNIST('mnist_data_sets', train=False, download=True,
                    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
                    batch_size=64, shuffle=True, **kwargs)


    for epoch in range(1, 6): # 5 epochen
        train(trainingsdata, model, epoch)
        test(testdata, model)



    #(un)comment preferred model
    t.save(model.state_dict(), 'persistent_models/Model1_sigmoid.pth')
    #t.save(model.state_dict(), 'persistent_models/Model2_relu.pth')

if __name__ == '__main__':
    main()
