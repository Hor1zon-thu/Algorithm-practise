import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets


epochs = 30
batch_size = 20

# prepare training and testing dataset
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# dataloader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)


# create the MLP network
class DigitMLP(torch.nn.Module):
    def __init__(self):
        super(DigitMLP, self).__init__()  #
        # with 2 hidden layers
        self.fc1 = nn.Linear(784, 512)  # 1st hidden layer
        self.fc2 = nn.Linear(512, 128)  # 2nd hidden layer
        self.fc3 = nn.Linear(128, 10)  # output layer
        # with only 1 hidden layer
        # self.fc1 = torch.nn.Linear(784, 30)
        # self.fc2 = torch.nn.Linear(30, 10)

    def forward(self, din):
        din = din.view(-1, 28 * 28)
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        dout = F.softmax(self.fc3(dout), dim=1)
        return dout


def trainDigit():
    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)  # define optimizer
    for epoch in range(epochs):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = lossfunc(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch: {}, Training Loss: {:.3f}'.format(epoch + 1, train_loss))
        testDigit()


def testDigit():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Test accuracy: %d %%' % (100 * correct / total))
    return 100.0 * correct / total


model = DigitMLP()

if __name__ == '__main__':
    trainDigit()