import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from Cross_Entropy_Loss import cross_entropy
import argparse
import os

class MLP(nn.Module):
    def __init__(self,img_size = 784,latent_1 = 512,latent_2 = 128,kinds_num=10):
        super().__init__()
        self.Linear1 = nn.Linear(img_size,latent_1)
        self.Linear2 = nn.Linear(latent_1,latent_2)
        self.Linear3 = nn.Linear(latent_2,kinds_num)
        self.dropuot = nn.Dropout(0.1)

    def forward(self,x):
        x = x.contiguous().view(-1,28*28) # x = torch.flatten(x, start_dim=1)
        x = F.relu(self.Linear1(x))
        x = self.dropuot(x)
        x = F.relu(self.Linear2(x))
        x = self.dropuot(x)
        x = self.Linear3(x)
        return x


def train(model, train_dataloader, test_dataloader, optimizer, loss_fuction, args,device):
    best_accuracy = 0.0

    for epoch in range(args.epoch):
        model.train()
        train_loss = 0
        for data,target in train_dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fuction(output,target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
        train_loss = train_loss / len (train_dataloader.dataset)
        print(f'Epoch: {epoch+1}/{args.epoch}, Training Loss: {train_loss:.4f}')

        test_accuracy = test(model, test_dataloader,device)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), './models/best_model.pth')
            print(f'Best model saved with accuracy: {best_accuracy:.4f}%')

    print(f'Finished Training. Best accuracy: {best_accuracy:.4f}%')

def test(model, test_dataloader,device):
    model.eval()
    correct = 0
    total =  0
    with torch.no_grad():
        for image,labels in test_dataloader:
            image, labels = image.to(device), labels.to(device)
            outputs = model(image)
            max_probability,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100.0 * correct / total
    print(f'Test accuracy: {test_acc:.2f}%')
    return test_acc


def main():
    parser = argparse.ArgumentParser(description="MNIST MLP Training")
    parser.add_argument('-e', '--epoch', type=int, default=100, help='训练轮数')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--cuda', default=True, help='是否使用GPU')
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    if not os.path.exists('./models'):
        os.makedirs('./models')


    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    train_data = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size,shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size,shuffle=False, num_workers=0)

    model = MLP()
    if args.cuda and torch.cuda.is_available():
        model = model.cuda()
    #loss_fuction = cross_entropy()
    loss_fuction = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    train(model, train_dataloader, test_dataloader, optimizer, loss_fuction, args,device)

'''
    model.load_state_dict(torch.load('best_model.pth'))
    if args.cuda and torch.cuda.is_available():
        model = model.cuda()
    final_accuracy = test(model, test_dataloader)
    print(f'Final Test Accuracy: {final_accuracy:.4f}%')
'''

if __name__ == '__main__':
    main()
