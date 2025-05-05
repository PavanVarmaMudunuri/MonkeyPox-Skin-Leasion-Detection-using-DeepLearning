import os
import shutil
import random
import torch
import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

torch.manual_seed(0)

class_names = ['chickenpox','Measles','monkeypox','normal']
root_dir = './Database'
source_dirs = ['chickenpox','Measles','monkeypox','normal']



            
class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs,transform):
        def get_images(class_name):                             '''Collects and returns all image filenames (ending with .jpg) from a specific class folder.'''
            print(class_name)
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('jpg')]
            print(f'Found {len(images)}{class_name}')
            return images
        self.images={}
        self.class_names=['chickenpox','Measles','monkeypox','normal']
        for c in self.class_names:
            self.images[c]=get_images(c)
        self.image_dirs=image_dirs
        self.transform=transform
                
    def __len__(self):                                                                    '''Returns the total number of images in the dataset across all classes.'''
        return sum([len(self.images[c]) for c in self.class_names])
    def __getitem__(self, index):                                               '''Picks a random class, gets the image at the given index, applies transformations,
                                                                                    and returns the image with its class label as an integer.'''
        class_name=random.choice(self.class_names)
        index=index%len(self.images[class_name])
        image_name=self.images[class_name][index]
        image_path =os.path.join(self.image_dirs[class_name], image_name)
        image=Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)


train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224,224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])


train_dirs = {
    'normal': 'Database/normal',
    'chickenpox': 'Database/chickenpox',
    'Measles': 'Database/Measles',
    'monkeypox': 'Database/monkeypox'
}
train_dataset=SkinDataset(train_dirs, train_transform)
test_dirs = {
    'normal': 'Database/test/normal',
    'chickenpox': 'Database/test/chickenpox',
    'Measles': 'Database/test/Measles',
    'monkeypox': 'Database/test/monkeypox',
    
}
test_dataset = SkinDataset(test_dirs, test_transform)

batch_size=6
dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print('Num of training batches', len(dl_train))
print('Num of test batches', len(dl_test))


class_names=train_dataset.class_names
def show_images(images, labels, preds):                '''Plots a batch of 6 images with actual (labels) and predicted (preds) class names.

                                                            Colors prediction label green if correct, red if incorrect.'''
    plt.figure(figsize=(8,4))
    for i, image in enumerate(images):
        plt.subplot(1,6,i+1, xticks=[], yticks=[])
        image=image.numpy().transpose((1,2,0))
        mean=np.array([0.485,0.456,0.406])
        std= np.array([0.229, 0.224, 0.225])
        image=image*std/mean
        image=np.clip(image,0.,1.)
        plt.imshow(image)
        col = 'green' if preds[i]==labels[i] else 'red'
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()
##########################
def show_images1(images, labels, preds):                           '''Similar to show_images but only shows
                                                                     predicted labels (both as x and y axis labels).'''
    plt.figure(figsize=(8,4))
    for i, image in enumerate(images):
        #plt.subplot(1,6,i+1, xticks=[], yticks=[])
        image=image.numpy().transpose((1,2,0))
        mean=np.array([0.485,0.456,0.406])
        std= np.array([0.229, 0.224, 0.225])
        image=image*std/mean
        image=np.clip(image,0.,1.)
        plt.imshow(image)
        col = 'green' if preds[i]==labels[i] else 'red'
        #plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.xlabel(f'{class_names[int(preds[i].numpy())]}', color=col)
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()

images, labels =next(iter(dl_train))
show_images(images, labels, labels)


images, labels =next(iter(dl_test))
show_images(images, labels, labels)

propNN =torchvision.models.resnet18(pretrained=True)
print(propNN)

propNN.fc=torch.nn.Linear(in_features=512, out_features=4)
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(propNN.parameters(), lr=3e-5)

def show_preds():                                 '''Fetches a test batch, runs the model on it, 
                                                    and visualizes predictions using show_images1.'''
    propNN.eval()
    images, labels =next(iter(dl_test))
    outputs = propNN(images)
    _, preds=torch.max(outputs, 1)
    show_images1(images, labels, preds)
#show_preds()

def train(epochs):                                    '''Trains the neural network for a given number of epochs. 
                                                   Performs: Training (forward pass, loss computation, backward pass, optimizer step). 
                                                   Validation every 20 steps, reporting accuracy and validation loss'''
            
    print('Starting training..')
    for e in range(0, epochs):
        print(f'Starting epoch {e+1}/{epochs}')
        print('='*20)
        train_loss=0
        propNN.train()
        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs=propNN(images)
            loss=loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss=loss.item()
            if train_step%20==0:
                print('Evaluating at step', train_step)
                acc=0.
                val_loss=0.
                propNN.eval()
                for val_step,(images, labels) in enumerate(dl_test):
                    outputs=propNN(images)
                    loss=loss_fn(outputs, labels)
                    val_loss+=loss.item()
                    _,preds=torch.max(outputs, 1)
                    acc+=sum(preds==labels).numpy()
                val_loss/=(val_step+1)
                acc=acc/len(test_dataset)
                print(f'Val loss: {val_loss:.4f}, Acc: {acc:.4f}')
                #show_preds()
                propNN.train()
                print('Accuracy:::',acc)
                
        train_loss/=(train_step+1)
        print(f'Training loss: {train_loss:.4f}')
train(epochs=1)
print('Result....')
show_preds()
