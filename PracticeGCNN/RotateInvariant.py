 # standard libraries

import torch.nn as nn # lets not write out torch.nn every time
import torch.nn.functional as F # functional versions of the modules in torch.nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from torchvision import transforms
from torchvision.transforms import ToPILImage
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp

x = torch.randn(1, 1, 28, 28) **2
r = 2

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Used Device: {device}")

def rotate(x: torch.Tensor, r: int) -> torch.Tensor:
# Method which implements the action of the group element `g` indexed by `r` on the input image `x`.
# The method returns the image `g.x`
# note that we rotate the last 2 dimensions of the input, since we want to later use this method to rotate minibatches containing multiple images
    return x.rot90(r, dims=(-2, -1))

def test():
    x = torch.randn(1, 1, 33, 33) ** 2

    r = 2
    gx = rotate(x, r)

    plt.imshow(x[0, 0].numpy())
    plt.title('Original Image $x$')
    plt.show()

    plt.imshow(gx[0, 0].numpy())
    plt.title('Rotated Image $g.x$')
    plt.show()


def text_plot_Example():
    filter3x3 = torch.randn(1, 1, 3, 3)
    print(filter3x3)
    plt.imshow(filter3x3[0, 0].numpy())
    plt.title('Filter')
    plt.show()
    gx = rotate(x, r)

    psi_x = torch.conv2d(x, filter3x3, bias=None, padding=1)
    psi_gx = torch.conv2d(gx, filter3x3, bias=None, padding=1)

    g_psi_x = rotate(psi_x, r)

    plt.imshow(g_psi_x[0, 0].numpy())
    plt.title('$g.\psi(x)$')
    plt.show()

    plt.imshow(psi_gx[0, 0].numpy())
    plt.title('$\psi(g.x)$')
    plt.show()

def plot_rotated_kernel(kernel: torch.Tensor, r: int):
    kernels = []
    loop_kernel = kernel
    fig, axs = plt.subplots(2, 2)

    for i in range(r):
         loop_kernel = torch.rot90(loop_kernel, k=1, dims=[2, 3])
         kernels.append(loop_kernel)

    for i in range(r):
         row = i // 2
         col = i % 2
         axs[row, col].imshow(kernels[i][0][0].detach().numpy(), cmap='gray')
         axs[row, col].set_title(f'{i * 90} degrees')
         axs[row, col].axis('off')

    plt.show()

class rotaClass(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, std=0.01):
        super(rotaClass, self).__init__()

        self.kernel_size = kernel_size
        self.std = std

        # define shared weights using nn.Parameter
        self.weight1 = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std,
                                    requires_grad=True)
        assert self.weight1.shape == (out_channels, in_channels, kernel_size, kernel_size)

        # plot_rotated_kernel(self.weight1, 4)
        self.bias1 = nn.Parameter(torch.zeros(out_channels), requires_grad=True)

        self.pool1 = nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1))

        # define four convolution layers with rotated kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=1)
        self.conv1.weight = nn.Parameter(self.weight1)
        self.conv1.bias = nn.Parameter(self.bias1)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.conv2.weight = nn.Parameter(
            torch.rot90(self.weight1, k=1, dims=[2, 3]))  # dims=[2, 3] rotates the second the third dimention / Rotate 90
        self.conv2.bias = nn.Parameter(self.bias1)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.conv3.weight = nn.Parameter(
            torch.rot90(self.weight1, k=2, dims=[2, 3]))  # dims=[2, 3] rotates the second the third dimention /Rotate 180

        self.conv3.bias = nn.Parameter(self.bias1)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.conv4.weight = nn.Parameter(
            torch.rot90(self.weight1, k=3, dims=[2, 3]))  # dims=[2, 3] rotates the second the third dimention. /Rotate 270
        self.conv4.bias = nn.Parameter(self.bias1)

    def forward(self, x):
        # apply four convolution layers separately
        x1 = self.conv1(x)
        x1 = nn.functional.relu(x1)

        x2 = self.conv2(x)
        x2 = nn.functional.relu(x2)

        x3 = self.conv3(x)
        x3 = nn.functional.relu(x3)

        x4 = self.conv4(x)
        x4 = nn.functional.relu(x4)

        # stack outputs and perform max pooling over new dimension
        x = torch.stack((x1, x2, x3, x4), dim=1)
        x, _ = torch.max(x, dim=1, keepdim=True)
        x_ = x.squeeze(2)
        assert torch.allclose(x, x_)

        # x = torch.flatten(x)
        # x = self.FC(x)
        # x = nn.functional.sigmoid(x)
        return x


class RotConvNet(nn.Module):
 def __init__(self, in_channels, out_channels, kernel_size=3, std=0.01):
     super(RotConvNet, self).__init__()

     self.kernel_size = kernel_size
     self.std = std

     # define shared weights using nn.Parameter
     self.weight1 = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std,
                                 requires_grad=True)
     assert self.weight1.shape ==(out_channels, in_channels, kernel_size, kernel_size)

     # plot_rotated_kernel(self.weight1, 4)
     self.bias1 = nn.Parameter(torch.zeros(out_channels), requires_grad=True)

     self.pool1 = nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1))

     # define four convolution layers with rotated kernels
     self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=1)
     self.conv1.weight = nn.Parameter(self.weight1)
     self.conv1.bias = nn.Parameter(self.bias1)

     self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
     self.conv2.weight = nn.Parameter(torch.rot90(self.weight1, k=1, dims=[2, 3])) # dims=[2, 3] rotates the second the third dimention / Rotate 90
     self.conv2.bias = nn.Parameter(self.bias1)

     self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
     self.conv3.weight = nn.Parameter(torch.rot90(self.weight1, k=2, dims=[2, 3])) # dims=[2, 3] rotates the second the third dimention /Rotate 180

     self.conv3.bias = nn.Parameter(self.bias1)

     self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
     self.conv4.weight = nn.Parameter(torch.rot90(self.weight1, k=3, dims=[2, 3])) # dims=[2, 3] rotates the second the third dimention. /Rotate 270
     self.conv4.bias = nn.Parameter(self.bias1)


     self.FC = nn.Linear(out_channels*32*32, out_channels)




 def forward(self, x):
     # apply four convolution layers separately
     x1 = self.conv1(x)
     x1 = nn.functional.relu(x1)

     x2 = self.conv2(x)
     x2 = nn.functional.relu(x2)

     x3 = self.conv3(x)
     x3 = nn.functional.relu(x3)

     x4 = self.conv4(x)
     x4 = nn.functional.relu(x4)

     # stack outputs and perform max pooling over new dimension
     x = torch.stack((x1, x2, x3, x4), dim=1)
     x, _ = torch.max(x, dim=1, keepdim=True)
     x_ = x.squeeze(2)
     assert torch.allclose(x, x_)

     x = torch.flatten(x)
     x = self.FC(x)
     x = nn.functional.sigmoid(x)
     return x


if __name__ == '__main__':
    mp.set_start_method('spawn')

    # Define transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    # Split dataset into training, validation, and test sets
    train_size = int(0.7 * len(trainset))
    val_size = int(0.15 * len(trainset))
    test_size = len(trainset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        trainset, [train_size, val_size, test_size])

    # Create data loaders for training, validation, and test sets
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                             shuffle=True, num_workers=2)

    # Define the model
    net = RotConvNet(in_channels=3, out_channels=10, kernel_size=3)
    net.to(device)
    output_dim = 10
    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
   # optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # Train the network
    for epoch in range(2):  # Loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(tqdm(trainloader)):
            # Get the inputs and labels
            inputs, labels_scalar = data[0].to(device), data[1].to(device)
            labels_scalar = labels_scalar.squeeze().int()
            labels = torch.zeros(10).to(device)
            labels[labels_scalar] = 1
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            outputs = outputs.view(-1, 10)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels_scalar).sum().item()
            if i % 2000 == 1999:  # Print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # Compute validation accuracy
        net.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data in tqdm(valloader):
                inputs, labels_scalar = data[0].to(device), data[1].to(device)
                labels_scalar = labels_scalar.squeeze().int()
                labels = torch.zeros(10).to(device)
                labels[labels_scalar] = 1
                outputs = net(inputs)
                outputs = outputs.view(-1, 10)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels_scalar).sum().item()
        val_acc = 100 * correct_val / total_val
        print('Epoch %d: Train loss: %.3f | Train acc: %.3f%% | Val acc: %.3f%%' %
              (epoch + 1, running_loss, 100 * correct / total, val_acc))

    print('Finished Training')

    # # Train the network
    # for epoch in range(2):  # Loop over the dataset multiple times
    #
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader):
    #         # Get the inputs and labels
    #         inputs, labels_scalar = data[0].to(device), data[1].to(device)
    #         labels_scalar = labels_scalar.squeeze().int()
    #         labels = torch.zeros(10).to(device)
    #         labels[labels_scalar]=1
    #         # Zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #         # Forward + backward + optimize
    #         outputs = net(inputs)
    #
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:    # Print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0
    #
    # print('Finished Training')
