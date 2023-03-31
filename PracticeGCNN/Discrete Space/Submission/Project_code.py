# standard libraries

import torch.nn as nn  # lets not write out torch.nn every time
import torch.nn.functional as F  # functional versions of the modules in torch.nn
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
import torch.nn.init as init
import numpy as np

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Used Device: {device}")


TRAIN = False

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


def text_plot_Example(x,r):
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

        # define shared weights using nn.Parameter
        self.weight1 = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        init.xavier_normal_(self.weight1)
        self.bias1 = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        assert self.weight1.shape == (out_channels, in_channels, kernel_size, kernel_size)

        # define four convolution layers with rotated kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=1)
        self.conv1.weight = nn.Parameter(self.weight1)
        self.conv1.bias = nn.Parameter(self.bias1)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.conv2.weight = nn.Parameter(
            torch.rot90(self.weight1, k=1,
                        dims=[2, 3]))  # dims=[2, 3] rotates the second the third dimention / Rotate 90
        self.conv2.bias = nn.Parameter(self.bias1)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.conv3.weight = nn.Parameter(
            torch.rot90(self.weight1, k=2,
                        dims=[2, 3]))  # dims=[2, 3] rotates the second the third dimention /Rotate 180

        self.conv3.bias = nn.Parameter(self.bias1)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.conv4.weight = nn.Parameter(
            torch.rot90(self.weight1, k=3,
                        dims=[2, 3]))  # dims=[2, 3] rotates the second the third dimention. /Rotate 270
        self.conv4.bias = nn.Parameter(self.bias1)

    def forward(self, x):
        # apply four convolution layers separately
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        # stack outputs and perform max pooling over new dimension
        x = torch.stack((x1, x2, x3, x4), dim=1)
        x, _ = torch.max(x, dim=1, keepdim=True)
        x = x.squeeze(dim=1)
        return x
    def plotWeights(self):
        plot_rotated_kernel(self.weight1, 4)

class RotConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, std=0.01):
        super(RotConvNet, self).__init__()

        self.kernel_size = kernel_size
        self.std = std
        self.dropout0 = nn.Dropout(p=0.2)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.3)

        self.Rot_1 = rotaClass(in_channels, out_channels*5, 3, std)
        self.BN_1 = nn.BatchNorm2d(out_channels*5)
        self.Rot_2 = rotaClass(out_channels*5, out_channels*20, 3, std)
        self.BN_2 = nn.BatchNorm2d(out_channels*20)
        self.Rot_3 = rotaClass(out_channels*20, out_channels*10, 3, std)
        self.BN_3 = nn.BatchNorm2d(out_channels*10)
        self.Rot_4 = rotaClass(out_channels*10, out_channels*4, 3, std)
        self.BN_4 = nn.BatchNorm2d(out_channels*4)
        self.Rot_5 = rotaClass(out_channels*4, out_channels, 3, std)
        self.BN_5 = nn.BatchNorm2d(out_channels)

        self.FC = nn.Linear(out_channels * 32 * 32, out_channels*6)
        self.FC3 = nn.Linear(out_channels*6 , out_channels)

    def forward(self, x):
        # apply four convolution layers separately

        x =self.Rot_1(x)
        # x =self.BN_1(x)
        x = nn.functional.relu(x)
        x= self.dropout0(x)
        x =self.Rot_2(x)
        x =self.BN_2(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)
        x =self.Rot_3(x)
        x =self.BN_3(x)
        x = nn.functional.relu(x)
        x =self.Rot_4(x)
        # x =self.BN_4(x)
        x = self.dropout3(x)
        x = nn.functional.relu(x)
        x =self.Rot_5(x)
        x =self.BN_5(x)
        x = nn.functional.relu(x)

        x = torch.flatten(x, 1)

        x = self.dropout1(x)
        x = nn.functional.relu(self.FC(x))
        # x= nn.functional.relu(self.FC2(x))
        x =nn.functional.relu(self.FC3(x))
        return x, F.softmax(x, dim=1)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == '__main__':
    mp.set_start_method('spawn')

    batch_size = 40
    output_dim = 10
    Num_epoch = 20
    LR = 0.0002
    in_channels = 3
    out_channels = 10
    kernel_size = 3
    ratio_T_T = 0.7

    # Define transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    # Split dataset into training, validation, and test sets
    train_size = int(ratio_T_T * len(trainset))
    val_size = int((1-ratio_T_T)/2 * len(trainset))
    test_size = len(trainset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        trainset, [train_size, val_size, test_size])

    # Create data loaders for training, validation, and test sets
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    # Define the model
    model = RotConvNet(in_channels=in_channels, out_channels=out_channels, kernel_size=in_channels)
    model.to(device)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    train_accuracy = []
    val_losses_list = []
    val_accuracy_list = []
    Test_losses = []
    Test_accuracy = []

    noisypicsdisplay = []
    predictedNoisy = []

    best_accuracy = 0.0

    if TRAIN:
        for epoch in range(Num_epoch):
            running_loss = 0.0
            T_correct = 0
            correct = 0
            total = 0

            # Train
            model.train()
            for i, data in enumerate(tqdm(trainloader,desc="Epoch {}/{}".format(epoch + 1, Num_epoch))):
                # Get the inputs and labels
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                digits, outputs = model(inputs)
                loss = criterion(digits, labels)
                _, predicted = torch.max(outputs.data, 1)

                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                total += labels.size(0)
                T_correct += (predicted == labels).sum().item()
                if i % (875-1) == 0 :
                    accuracy = 100 * T_correct / total
                    print('[%d, %5d] loss: %.3f, accuracy: %.2f %%' %
                          (epoch + 1, i + 1, running_loss / len(trainloader), accuracy))

            train_loss = running_loss / len(trainloader)
            train_losses.append(train_loss)
            train_accuracy.append(100 * T_correct / total)

            model.eval()
            val_loss = 0
            total = 0
            val_accuracy = 0
            with torch.no_grad():
                for data in tqdm(valloader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    total += labels.size(0)

                    outputs = model(inputs)[1]
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_accuracy += (predicted == labels).sum().item()

                val_loss /= len(valloader)
                # val_accuracy /= len(valloader)

                val_losses_list.append(val_loss)
                val_accuracy_list.append(100 * val_accuracy / total)


            print('Test Accuracy of the model on the 10000 test images: {:.4f} %'.format(100 * val_accuracy / total))

            # Evaluation
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            correct_noisy = 0
            total_noisy = 0
            with torch.no_grad():
                for data in tqdm(testloader):
                    inputs, labels = data[0].to(device), data[1].to(device)

                    outputs = model(inputs)[1]
                    test_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    outputs_Noisy = model(inputs)[1]
                    test_loss += criterion(outputs_Noisy, labels).item()
                    _, predicted_noisy = torch.max(outputs_Noisy.data, 1)
                    total_noisy += labels.size(0)
                    correct_noisy += (predicted_noisy == labels).sum().item()

                test_loss /= len(testloader)
                Test_losses.append(test_loss)
                Test_accuracy.append(100 * correct_noisy / total)

                if epoch % 3 == 0:
                    noisypicsdisplay.append(inputs[0, 0, :, :])
                    predictedNoisy.append(predicted_noisy[0])

                print('Test Accuracy of the model on the 10000 test images: {:.4f} %'.format( 100*correct / total))

            # Save model parameters if the current accuracy is better than the best one so far
            current_accuracy = 100 * correct / total
            if current_accuracy > best_accuracy:
                print("Parameters Mode Updated")
                best_accuracy = current_accuracy
                torch.save(model.state_dict(), "best_model.pth")

        print('Finished Training')

        # Plot the training and validation loss/accuracy after each epoch
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.plot(train_losses, label="Training Loss")
        ax1.plot(val_losses_list, label="Validation Loss")

        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2.plot(train_accuracy, label="Training Accuracy")
        ax2.plot(val_accuracy_list, label="Validation Accuracy")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        plt.tight_layout()

        # Save the plot to a file
        plt.savefig(f"epoch_{epoch+1}.png")
        plt.close()

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
                                             shuffle=True, num_workers=2)

    #evaluate

    # Set the model to evaluation mode
    model.load_weights("best_model.pth")
    model.eval()

    # Initialize lists to store correct and incorrect predictions
    correct_predictions = []
    incorrect_predictions = []

    # Loop through the test data and make predictions
    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)[1]
            _, predicted = torch.max(outputs.data, 1)
            correct_mask = (predicted == labels)

            # Append correct predictions to the correct_predictions list
            for i in range(len(labels)):
                if correct_mask[i]:
                    if len(correct_predictions) < 10:
                        correct_predictions.append((inputs[i], predicted[i], labels[i]))

            # Append incorrect predictions to the incorrect_predictions list
            for i in range(len(labels)):
                if not correct_mask[i]:
                    if len(incorrect_predictions) < 10:
                        incorrect_predictions.append((inputs[i], predicted[i], labels[i]))

    # Plot the correct predictions
    fig, axs = plt.subplots(2, 5, figsize=(13, 8))
    fig.suptitle('Correct Predictions')
    for i, (image, predicted_label, true_label) in enumerate(correct_predictions):
        row = i // 5
        col = i % 5
        axs[row, col].imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
        axs[row, col].set_title(f'Predicted: {predicted_label}, True: {true_label}')
        axs[row, col].axis('off')

    fig, axs = plt.subplots(5, 5, figsize=(13, 8))
    fig.suptitle('Correct Classification-Rotated Images')
    amount_of_pics = 5
    for i, (image, predicted_label, true_label) in enumerate(correct_predictions[:amount_of_pics]):
        for j in range(4):
            axs[i, 0].imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
            rotated_image = rotate(image, j )
            output_field, output = model((rotated_image.unsqueeze(0)))
            _, predicted = torch.max(output.data, 1)
            output_field = output_field.squeeze(0)
            row = i
            col = j+1
            axs[row, col].imshow(output_field.cpu().detach().numpy().reshape(2, 5))
            if(row==0):
                axs[0, col].set_title(f'Field Rotation: {j * 90} deg.\nPredicted: {predicted.item()}, True: {true_label}')
            else:
                axs[row, col].set_title(
                    f'Predicted: {predicted.item()}, True: {true_label}')
            axs[row, col].axis('off')

    fig, axs = plt.subplots(5, 5, figsize=(13, 8))
    fig.suptitle('Incorrect Classification-Rotated Images')
    for i, (image, predicted_label, true_label) in enumerate(incorrect_predictions[:amount_of_pics]):
        for j in range(4):
            axs[i, 0].imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
            rotated_image = rotate(image, j )
            output_field, output = model((rotated_image.unsqueeze(0)))
            _, predicted = torch.max(output.data, 1)

            output_field = output_field.squeeze(0)
            row = i
            col = j+1
            axs[row, col].imshow(output_field.cpu().detach().numpy().reshape(2, 5))
            if(row==0):
                axs[0, col].set_title(f'Field Rotation: {j * 90} deg.\nPredicted: {predicted.item()}, True: {true_label}')
            else:
                axs[row, col].set_title(
                    f'Predicted: {predicted.item()}, True: {true_label}')

            axs[row, col].axis('off')

    # Plot the incorrect predictions
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle('Incorrect Predictions')
    for i, (image, predicted_label, true_label) in enumerate(incorrect_predictions):
        row = i // 5
        col = i % 5
        axs[row, col].imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
        axs[row, col].set_title(f'Predicted: {predicted_label}, True: {true_label}')
        axs[row, col].axis('off')

    plt.show()


    print('Done')

