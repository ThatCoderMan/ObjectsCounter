import os
import json
import numpy as np
import skimage.draw
import cv2

IMAGE_FOLDER = "./train/"
MASK_FOLOER = "./mask/"
PATH_ANNOTATION_JSON = 'Json'

# Загрузить VIA экспортированный файл JSON
annotations = json.load(open(PATH_ANNOTATION_JSON, 'r'))
imgs = annotations["_via_img_metadata"]

for imgId in imgs:
    filename = imgs[imgId]['filename']
    regions = imgs[imgId]['regions']
    if len(regions) <= 0:
        continue

         # Удалить первую отмеченную категорию, в этом примере отмечен только объект
    polygons = regions[0]['shape_attributes']

         # Путь к изображению
    image_path = os.path.join(IMAGE_FOLDER, filename)
         # Прочитайте картинку, целью которой является получение информации о ширине и высоте
    image = cv2.imread(image_path)  # image = skimage.io.imread(image_path)
    height, width = image.shape[:2]

         # Создать пустую маску
    maskImage = np.zeros((height,width), dtype=np.uint8)
    countOfPoints = len(polygons['all_points_x'])
    points = [None] * countOfPoints
    for i in range(countOfPoints):
        x = int(polygons['all_points_x'][i])
        y = int(polygons['all_points_y'][i])
        points[i] = (x, y)

    contours = np.array(points)

         # Пересечь все координаты картинки
    for i in range(width):
        for j in range(height):
            if cv2.pointPolygonTest(contours, (i, j), False) > 0:
                maskImage[j,i] = 1

    savePath = MASK_FOLOER + filename
         # Сохранить маску
    cv2.imwrite(savePath, maskImage)