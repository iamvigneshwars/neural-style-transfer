import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import copy



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Desired image size
imsize = 512
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  
    transforms.ToTensor()])

def image_loader(image_name):

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)

    return image.to(device, torch.float)


style_img = image_loader("picasso.jpg")
content_img = image_loader("dancer.jpg")
