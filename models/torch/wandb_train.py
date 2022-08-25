import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision, wandb
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import *
from torchsummary import summary
from tqdm import tqdm

# cifar10
batch_size = 64
num_train = 10000

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainset = torch.utils.data.Subset(trainset, range(num_train))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           

# model set up
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
summary(model, (3, 32, 32))

# init wandb
wandb.init(project="cifar-10-grid")
config = wandb.config

# hyperparamters
config.batch_size = batch_size
config.epochs = 20
config.architecture = "vanilla resnet34"
config.activation = "softmax"
config.latent = 512

# training 
for epoch in range(20): 
    with tqdm(trainloader, unit="batch") as tepoch:

        running_loss = 0.0
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            outputs = model(data).to(device)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # calculating accuracy
            predictions = outputs.argmax(dim=1, keepdim=True)
            correct = (predictions == target).sum().item()
            accuracy = correct / batch_size

            # stats
            tepoch.set_postfix(loss=loss.item(), accuracy=100.*accuracy)

        # log
        wandb.log({
            "epoch": epoch,
            "test_accuracy": 100.*accuracy,
            "test_loss": loss.item()
        })

wandb.finish()

# saving
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

# testing
dataiter = iter(testloader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)

# print images
plt.figure(figsize=(20,20))
plt.imshow(torchvision.utils.make_grid(images.cpu()).permute(1, 2, 0))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))