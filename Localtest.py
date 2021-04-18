# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:11:12 2021

@author: Uni361004
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
import random
import os

class MyDataset():
  def __init__(self, dset_dir, transforms=T.Compose([])):
      
    self.dset_dir = Path(dset_dir)
    self.transforms = transforms
    self.files = []
    
    folders = sorted(os.listdir(self.dset_dir))
    for folder in folders:
      class_idx = folders.index(folder)
      folder_dir = self.dset_dir/folder
      files = os.listdir(folder_dir)
      self.files += [{"file": folder_dir/x, "class": class_idx} for x in files]

  def __len__(self):
    return len(self.files)

  def __getitem__(self, i):
    item = self.files[i]
    file = item['file']
    
    class_idx = torch.tensor(item['class'])
    
    img = Image.open(file).convert("RGB")
    img = self.transforms(img)
    return img, class_idx

class MyCNN(nn.Module):
  def __init__(self, use_norm=False):
    super().__init__()
    self.convolution = nn.Sequential(
      # 1
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      # 2
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      # 3
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      # 4
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      # 5
      nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      # 6
      nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1, stride=2),
      nn.ReLU(),
      nn.AdaptiveMaxPool2d(output_size=2),
    )
    self.classifier = nn.Sequential(
        nn.Linear(8192, 8192),
        nn.ReLU(),
        nn.Linear(8192,8)
    )

  def forward(self, x):
    b, _, _, _ = x.shape
    feature_extracted = self.convolution(x) 
    #return feature_extracted
    #return feature_extracted.view(b, -1)
    return self.classifier(feature_extracted.view(b, -1))


def train(net, loaders, optimizer, criterion, epochs=100, dev=torch.device('cpu')):
    try:
        net = net.to(dev)
        print(net)

        history_loss = {"train": [], "val": [], "test": []}
        history_accuracy = {"train": [], "val": [], "test": []}

        for epoch in range(epochs):
            print("----------------------------------------------------------------------")
            print(f"Epoch {epoch+1} ...")

            sum_loss = {"train": 0, "val": 0, "test": 0}
            sum_accuracy = {"train": 0, "val": 0, "test": 0}

            for split in ["train", "val", "test"]:
                if split == "train":
                  net.train()
                else:
                  net.eval()

                for (input, labels) in loaders[split]:

                    input = input.to(dev)
                    labels = labels.to(dev)

                    optimizer.zero_grad()

                    pred = net(input)
                    loss = criterion(pred, labels)

                    sum_loss[split] += loss.item()

                    if split == "train":

                        loss.backward()

                        optimizer.step()

                    _ , pred_labels = pred.max(1)
                    batch_accuracy = (pred_labels == labels).sum().item()/input.size(0)

                    sum_accuracy[split] += batch_accuracy

            epoch_loss = {split: sum_loss[split]/len(loaders[split]) for split in ["train", "val", "test"]}
            epoch_accuracy = {split: sum_accuracy[split]/len(loaders[split]) for split in ["train", "val", "test"]}

            for split in ["train", "val", "test"]:
                history_loss[split].append(epoch_loss[split])
                history_accuracy[split].append(epoch_accuracy[split])

            print(f"TrL = {epoch_loss['train']:.4f},",
                  f"TrA = {epoch_accuracy['train']:.4f},",
                  f"VL = {epoch_loss['val']:.4f},",
                  f"VA = {epoch_accuracy['val']:.4f},",
                  f"TeL = {epoch_loss['test']:.4f},",
                  f"TeA = {epoch_accuracy['test']:.4f}")
    
    except KeyboardInterrupt:
        print("***********************")
        print("***** Interrupted *****")
        print("***********************")
    
    finally:
        # Plot loss
        plt.title("Loss")
        for split in ["train", "val", "test"]:
            plt.plot(history_loss[split], label=split)
        plt.legend()
        plt.show()
        # Plot accuracy
        plt.title("Accuracy")
        for split in ["train", "val", "test"]:
            plt.plot(history_accuracy[split], label=split)
        plt.legend()
        plt.show()
        
if __name__=='__main__':
    
    dest_dir='C:\\Users\\emanu\\Desktop\\project_data\\data'
    os.listdir(dest_dir)
    num_classes = len(os.listdir(dest_dir))
    print(f"Number of classes: {num_classes}")
    
    transforms = T.Compose([
        T.Resize((300,300)),
        T.RandomCrop((224,224)),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
        T.RandomHorizontalFlip(p=0.4), 
        T.RandomVerticalFlip(p=0.4)
    ])
    
    data = MyDataset(dest_dir,transforms=transforms)
    
    # example, label = data[1001]
    # plt.imshow(  example.permute(1, 2, 0)  )
    
    n = len(data)
    idx = list(range(n))
    random.seed(0)
    random.shuffle(idx)
    test_frac = 0.1
    num_test = int(n * test_frac) 
    num_train = n - 2*num_test
    
    train_idx = idx[:num_train]
    val_idx = idx[num_train : num_train+num_test]
    test_idx = idx[num_train+num_test :]
    
    print(f"{n} samples")
    print(f"{len(train_idx)} samples used as train set")
    print(f"{len(val_idx)} samples used as validation set")
    print(f"{len(test_idx)} samples used as test set")
    
    train_set = Subset(data, train_idx)
    val_set = Subset(data, val_idx)
    test_set = Subset(data, test_idx)
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True, drop_last=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, drop_last=False, num_workers=2)
    
    loaders = {"train": train_loader,
               "val" : val_loader,
               "test": test_loader}
    
    net = MyCNN()

    x = torch.rand(8, 3, 256, 256)
    out = net(x)
    print(f'output shape: {out.shape}')
            
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if torch.cuda.is_available(): torch.cuda.empty_cache() 
    
    optimizer = optim.SGD(net.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    train(net, loaders, optimizer, criterion, dev=device)
