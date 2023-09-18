import numpy as np
import config
import os
from PIL import Image, ImageFile
import cv2 
from torchvision.utils import save_image
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math

def split_image_crops(dir, model, kernel_size = 256, device='cpu'):
    model = model.to(device)

    for idx, image_file in enumerate(os.listdir(dir)):
        image = Image.open(os.path.join(dir, image_file)).convert('RGB')
        width, height = image.size
        max_size = math.ceil(max(height, width) / kernel_size) * kernel_size
        pad_height = max_size -height
        pad_width = max_size - width

        image = np.array(image)
        augment = A.compose([
            A.PadIfNeeded(min_height=max_size, min_width=max_size, border_mode=cv2.BORDER_REFLECT),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
            ToTensorV2()
        ])

        image = augment(image=image)['image'].to(device)
        img_size = image.shape[2]
        image = image.permute(1,2,0)
        kh, hw = kernel_size, kernel_size
        dh, dw = 32, 32

        patches = image.unfold()


