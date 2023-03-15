# %%
import os

from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets
import torch
import PIL
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
batch_size = 64
number_of_labels = 42
learning_rate = 0.1
num_epochs = 150
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
    class_encoder[classes[i]]=i
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir,self.img_labels[idx])
        image = PIL.Image.open(img_path)
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
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
    transforms.Resize((32,32))
])


full_dataset = torchvision.datasets.ImageFolder("/home/e.sofronov/cnn_simpsons/characters",transformations)
train_dataset,valid_dataset = torch.utils.data.random_split(full_dataset,[0.7, 0.3])
train_dataset, test_set = torch.utils.data.random_split(full_dataset,[0.8, 0.2])
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2)
print("The number of images in a training set is: ", len(train_loader)*batch_size)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
print("The number of images in a test set is: ", len(test_loader)*batch_size)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
print("The number of images in validation set is: ",len(valid_loader)*batch_size)
print("The number of batches per epoch is: ", len(train_loader))


# %%
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Sequential( 
                nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        )                      
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Sequential( 
                nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        )                      
        #self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Sequential( 
                nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        )  
        #self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Sequential( 
                nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        )  
        #self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*16*16, 42)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 24*16*16)
        output = self.fc1(output)

        return output

# Instantiate a neural network model 
model = Network().to(device)

# %%
from torch.optim import Adam
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=3, verbose=True, threshold=1e-2
)

# %%
from torch.autograd import Variable
import tqdm
# Function to save the model
def saveModel():
    path = "./simpsons.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
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
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)

def train():
    
    best_accuracy = 0.0

    # Define your execution device
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    for epoch in tqdm.trange(num_epochs):  # loop over the dataset multiple times
        losses = []

        for i, (images, labels) in enumerate(train_loader, 0):

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
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

# %%
import torchmetrics
# Function to test what classes performed well
def testClassess():
    class_correct = list(0. for i in range(number_of_labels))
    class_total = list(0. for i in range(number_of_labels))
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=42,average=None).to(device)
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            acc = metric(predicted, labels)
    for i in range(number_of_labels):
        print(f'Accuracy of {classes[i]} : {acc[i]}')

# %%
if __name__ == "__main__":
    
    train()
    print('Finished Training')
    testClassess()
    model = Network()
    path = "simpsons.pth"
    model.load_state_dict(torch.load(path))
    #testBatch()


