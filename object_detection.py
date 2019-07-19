import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/nik/Pictures/canada-mountain-road-wallpaper-61832-63678-hd-wallpapers.jpg', 1)
print(img.shape)
plt.imshow(img[:, :, ::-1])
plt.show()
