import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = r"F:\XLTHS\Homework\Bai15\lena.jpg"


lena = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

plt.imshow(lena, cmap="gray")
plt.title("Ảnh Lena gốc")   
plt.axis("off")
plt.show()
