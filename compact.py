#!/bin/env python3

# %%
# Unet model implementation
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from PIL import Image
import os

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
#Test code with generated data for architecture testing
"""
def test_a():
    x = torch.randn((3, 1, 256, 256))    
    print(x.dtype)
    print(x.size)
    print("dimenze:" + str(x.ndim))
    print(x.shape)
    
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape 
"""
#Test code with preprocessed data for architecture testing
"""
def test_b():
    PATH_TO_IMAGES = 'C:\\Users\\Dell\\Desktop\\skola\\rocnik_3\\bakalarka\\implementace\\github_implementace\\CNN_UNet\\train_img\\train2.txt'
    obr = np.loadtxt(PATH_TO_IMAGES, dtype=np.float)
    obr = obr.reshape (1, 1, 256, 256).astype('float32')
    obr = torch.from_numpy(obr)
    print("testb dtype::"+str(obr.dtype))
    print("testb size:"+str(obr.size))
    print("testtb ndim:"+str(obr.ndim))
    print("testtb shape:"+str(obr.shape))      

    model = UNET(in_channels=1, out_channels=1)
    preds = model(obr)

    yz = preds.detach().numpy()
    print()
    print(yz.shape)

    arr = yz
    arr_ = np.squeeze(arr)
    plt.imshow(arr_)
    plt.show()

    
if __name__ == "__main__":
    test_a()
    test_b()
"""

# %%
# Loader for data
import os
import numpy as np
#from PIL import Image
from torch.utils.data import Dataset


class MicroscopyDataset (Dataset):
    
    """Loading data fuction"""
    
    def __init__(self, image_dir, mask_dir, transform=None):
        print("Debug: -here-")
        
        #print("Path to image:" + str(image_dir))
        #print("Path to mask:" + str(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir   
        #list of all files in folder
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        print("Length: -here-")
        
        #Length of the dataset
        return len(self.images)
    
    def __getitem__(self,idx):
        print("Get: -here-")
        
        #Path for image and mask
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        print(str(image_path))
        print(str(mask_path))
        
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        image = image.reshape(3, 256, 256).astype('float32') 
        mask = mask.reshape(1, 256, 256).astype('float32')
        mask[mask == 255.0] = 1.0
        return image, mask
        


# %%
# utility helpers (not all yet used TODO)
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    train_transform,
    val_transform,
    batch_size,
    num_workers=4,
    pin_memory=True):
    
    train_ds = MicroscopyDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = MicroscopyDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    print(type(loader))
    print("Breakpoint!")
    with torch.no_grad():
        for x, y in loader:
            x = x
            y = y
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

# %%
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

TRAIN_IMG_DIR = 'train_img/'
TRAIN_MASK_DIR = 'train_mask/'
VAL_IMG_DIR = 'val_img/'
VAL_MASK_DIR = 'val_mask/'

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 1
NUM_WORKERS = 0
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        train_dir = TRAIN_IMG_DIR,
        train_maskdir = TRAIN_MASK_DIR,
        val_dir = VAL_IMG_DIR,
        val_maskdir = VAL_MASK_DIR,
        batch_size = BATCH_SIZE,
        train_transform = train_transform,
        val_transform = val_transforms,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()

# %%
