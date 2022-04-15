import torch
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True      ## solve error "OSError: image file is truncated"
Image.MAX_IMAGE_PIXELS = None   ## solve error image with significant pixels


device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



## transform operations to load image, the load data range [0,1]
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])



## load data

def load_data(data_path=None,transform=None,batch_size=16,shuffle=True,num_workers=2):
    dataset = datasets.ImageFolder(data_path,transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)

    return dataloader


batch_size = 16
traindata_path = '../data/image_split/train/'
trainloader = load_data(data_path=traindata_path,
                        transform=transform,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=2)

testdata_path = '../data/image_split/test/'
testloader = load_data(data_path=testdata_path,
                        transform=transform,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=2)





import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.to(device)


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device),data[1].to(device)

        # forward + backward + optimize
        outputs = net(inputs)
        # zero the parameter gradients
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 20 == 19:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the  test images: %d %%' % (
    100 * correct / total))