import glob
from PIL import Image

imgs = glob.glob('data/dataset/masks/*.png')
for img in imgs:
    image = Image.open(img)
    image = image.convert('RGB')
    image.save(img, format='PNG', quality=100)
