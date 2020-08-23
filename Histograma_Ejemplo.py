#////////////////// Repositorio en GitHub por: Adriel Machado //////////////////////////////////
#La persona que sabe manejar el "histograma" sabe diseñar un buen sistema de vision artificial
#Version 1.0 de histograma con cv2
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Cargar imagen
grayImg = cv2.imread('lenna.png', 0) # Se carga la imagen en escala de grises
CMin = 100  # Se define el valor mínimo deseado del histograma 
CMax = 200  # Se define el valor máximo deseado del histograma
minVal = np.amin(grayImg)  # Se calcula pixel con máximo valor en la imagen de entrada
maxVal = np.amax(grayImg)  # Se calcula pixel con minimo valor en la imagen de entrada
hist, bins = np.histogram(grayImg.flatten(), 256, [0, 256])  # Se crea el objeto de istograma
height, width = grayImg.shape[:2]  # Obtenemos sus dimensiones
img2 = np.zeros((height, width), np.uint8)  # Creamos una imagen nueva

for i in range(0, height):
    for j in range(0, width):
        img2[i, j] = ((CMax - CMin) / (maxVal - minVal)) * (grayImg[i, j] - minVal) + CMin
        # Se aplica la ecuación de reducción iterativamente a cada pixel

# cv2.imshow('images',  np.hstack([grayImg, img2]))  # Se compara la imagen original con la alterada
# cv2.waitKey(0)
# cv2.destroyAllWindows()

hist = cv2.calcHist([img2], [0], None, [256], [0, 256])  # Se calcula el histograma de la imagen contraída
plt.subplot(221), plt.imshow(img2, cmap=plt.get_cmap('gray'))
plt.title('Imagen gris')
plt.subplot(222), plt.hist(img2.ravel(), 256, [0, 256])
plt.title('Histograma para imagen gris')
plt.show()
