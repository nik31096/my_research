import torch
from torch import nn
import cv2
from matplotlib import pyplot as plt
from copy import deepcopy

img = cv2.imread('/home/nik-96/Pictures/wallpaper1.jpg', 1)[:, :, ::-1]

print(img.shape)

plt.imshow(img)
plt.show()

m1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 3), stride=1)
m2 = nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=(3, 3), stride=1)
m2.weight = deepcopy(m1.weight)
imgTensor = torch.tensor([img], dtype=torch.float32).permute(0, 3, 1, 2)
print(imgTensor.shape)
x = m1(imgTensor)
newImg = m2(x)[0].permute(1, 2, 0).data.numpy()
print(newImg.shape)
plt.imshow(newImg)
plt.show()


