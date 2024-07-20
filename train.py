from efficientnet_pytorch import EfficientNet
from torch import nn
import torch.optim as optim
from torchvision import transforms,  datasets
import torch
from torch.utils.data import DataLoader
from tqdm import *

train_dir = "./train"
val_dir = "./test"
epoch = 100
batch_size = 32
lr = 0.01


#net = EfficientNet.from_pretrained('efficientnet-b3')
#net._fc.out_features = 376
net = torch.load("model3.pth")
net.cuda()


train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation((0,45)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])])
test_transforms = transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

train_datasets=datasets.ImageFolder(train_dir,transform=train_transforms)
train_dataloaders= DataLoader(train_datasets,batch_size=batch_size,shuffle=True)
train_dataset_sizes = len(train_datasets)
val_datasets=datasets.ImageFolder(val_dir,transform=test_transforms)
val_dataloaders= DataLoader(val_datasets,batch_size=1)
val_dataset_sizes = len(val_datasets)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters(), 1e-3,betas=(0.9, 0.999), eps=1e-8,weight_decay=0,amsgrad=False)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

best_acc = 0.0
for i in range(epoch):
    net.train()
    running_loss = 0.0
    for images, labels in tqdm(train_dataloaders,desc=f"Epoch:{i+1}"):
        optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for x, y in tqdm(val_dataloaders):
            x,y = x.cuda(), y.cuda()
            outputs = net(x)
            predicted = torch.max(outputs, dim=1)[1]
            acc += (predicted == y).sum().item()
    accurate = acc / val_dataset_sizes
    train_loss = running_loss / train_dataset_sizes
    print(f"acc:{accurate},loss:{train_loss}")
    if best_acc<accurate:
        best_acc = accurate
        torch.save(net,"model.pth")
