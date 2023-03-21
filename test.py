import os

from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import SGD
from torch.autograd import Variable
import tqdm
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
number_of_labels = 42
learning_rate = 0.001
num_epochs = 100
classes = ('abraham_grampa_simpson',
           'agnes_skinner',
           'apu_nahasapeemapetilon',
           'barney_gumble',
           'bart_simpson',
           'carl_carlson',
           'charles_montgomery_burns',
           'chief_wiggum',
           'cletus_spuckler',
           'comic_book_guy',
           'disco_stu',
           'edna_krabappel',
           'fat_tony',
           'gil',
           'groundskeeper_willie',
           'homer_simpson',
           'kent_brockman',
           'krusty_the_clown',
           'lenny_leonard',
           'lionel_hutz',
           'lisa_simpson',
           'maggie_simpson',
           'marge_simpson',
           'martin_prince',
           'mayor_quimby',
           'milhouse_van_houten',
           'miss_hoover',
           'moe_szyslak',
           'ned_flanders',
           'nelson_muntz',
           'otto_mann',
           'patty_bouvier',
           'principal_skinner',
           'professor_john_frink',
           'rainier_wolfcastle',
           'ralph_wiggum',
           'selma_bouvier',
           'sideshow_bob',
           'sideshow_mel',
           'snake_jailbird',
           'troy_mcclure',
           'waylon_smithers')
class_encoder = {}
for i in range(len(classes)):
    class_encoder[classes[i]] = i


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path)
        label = self.img_labels[idx]
        class_indicator = label.rfind('_')
        class_str = label[:class_indicator]
        label = class_encoder[class_str]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Loading and normalizing the data.
# Define transformations for the training and test sets
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    transforms.Resize((32, 32), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
])

full_dataset = torchvision.datasets.ImageFolder("./characters", transformations)
_, valid_dataset = torch.utils.data.random_split(full_dataset, [0.7, 0.3])
train_dataset, test_set = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        self.fc1 = nn.Linear(512 * 2 * 2, 42)

    def forward(self, input_img):
        output = F.relu(self.conv1(input_img))
        output = F.relu(self.conv2(output))
        output = self.pool(output)
        output = F.relu(self.conv4(output))
        output = F.relu(self.conv5(output))
        output = self.pool1(output)
        output = F.relu(self.conv6(output))
        output = F.relu(self.conv7(output))
        output = self.pool2(output)
        output = F.relu(self.conv8(output))
        output = F.relu(self.conv9(output))
        output = self.pool3(output)
        output = F.relu(self.conv10(output))
        output = output.view(-1, 512 * 2 * 2)
        output = self.fc1(output)
        return output


# Instantiate a neural network model
model = Network().to(device)

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=3, verbose=True, threshold=1E-2
)


# Function to save the model
def saveModel():
    path = "./simpsons.pth"
    torch.save(model.state_dict(), path)


# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    metric = torchmetrics.Recall(task="multiclass", num_classes=42).to(device)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
            metric(predicted, labels)
    recall = metric.compute()
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)

    return accuracy, recall

loss_metric =[]
recall_metric=[]
accuracy_metric=[]
lr_metric=[]
def train():
    best_accuracy = 0.0

    # Define your execution device
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    for epoch in tqdm.trange(num_epochs, position=0, desc="Epochs", leave=True):  # loop over the dataset multiple times
        losses = []

        for i, (images, labels) in enumerate(tqdm.tqdm(train_loader, position=1, desc="Batch iter", leave=False), 0):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()
        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)
        print(f"Loss at epoch {epoch} = {mean_loss}")
        # Compute and print the average accuracy fo this epoch when tested over all test images
        accuracy, recall = testAccuracy()
        loss_metric.append(mean_loss)
        accuracy_metric.append(accuracy)
        recall_metric.append(recall)
        print(f"For epoch {epoch} recall: {recall}")
        print(f'For epoch {epoch} the test accuracy over the whole test set is {accuracy} %')

        # we want to save the model if the accuracy is the best
        if recall > best_accuracy:
            saveModel()
            best_accuracy = recall


# Function to test what classes performed well
def test_classes():
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=42, average=None).to(device)
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            metric(predicted, labels)
    acc = metric.compute()
    for i in range(number_of_labels):
        print(f'Accuracy of {classes[i]} : {acc[i]}')


# import torchinfo


# torchinfo.summary(model, depth=2, input_size=(32, 3, 32,32), row_settings=["var_names"], verbose=0, col_names=[
# "input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"])


# Function to show the images
def image_show(img):
    img = img / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def test_batch():
    # get batch of images from the test DataLoader
    images, labels = next(iter(valid_loader))

    # show all images as one image grid
    image_show(torchvision.utils.make_grid(images))
    images = images.to(device)
    # Show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
                                    for j in range(batch_size)))

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(batch_size)))


if __name__ == "__main__":
    train()
    print('Finished Training')
    model = Network().to(device)
    path = "simpsons.pth"
    model.load_state_dict(torch.load(path))
    test_classes()
    test_batch()
    epochs = [i for i in range(num_epochs)]
    fig, ax = plt.subplots()
    ax.plot(recall_metric, epochs, label="Recall")
    ax.plot(loss_metric, epochs, label="Loss")
    ax.plot(accuracy_metric, epochs, label="Accuracy")
    plt.legend()
    plt.show()