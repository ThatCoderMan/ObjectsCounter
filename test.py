from skimage.io import imread
import matplotlib.pyplot as plt

image = 'data/dataset/images/frame_0001.jpg'
mask = 'data/dataset/masks/frame_0001.png'
image = imread(image)
mask = imread(mask)
# mask = resize(mask, (mask.shape[0], mask.shape[1]))
print(mask)
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(2, 3), dpi=125)
ax[0].set_title('Image')
ax[0].set_axis_off()
ax[0].imshow(image)
ax[1].set_title('Mask')
ax[1].set_axis_off()
ax[1].imshow(mask * 100)
plt.show()
plt.close()
