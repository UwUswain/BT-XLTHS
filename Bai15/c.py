import cv2
import numpy as np
import matplotlib.pyplot as plt

# (a) Load ảnh Lena
img_path = r"F:\XLTHS\Homework\Bai15\lena.jpg"
lena = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if lena is None:
    raise FileNotFoundError(f"❌ Không tìm thấy file: {img_path}")

plt.figure(figsize=(6,6))
plt.imshow(lena, cmap='gray')
plt.title("Ảnh Lena gốc")
plt.axis("off")
plt.show()

# (b) Kernel trung bình 3x3
h1 = np.ones((3,3), np.float32) / 9
lena_filtered_3x3 = cv2.filter2D(lena, -1, h1, borderType=cv2.BORDER_CONSTANT)

plt.figure(figsize=(6,6))
plt.imshow(lena_filtered_3x3, cmap='gray')
plt.title("Ảnh Lena sau khi lọc (3x3 mean filter)")
plt.axis("off")
plt.show()

# (c) Kernel trung bình 5x5
h2 = np.ones((5,5), np.float32) / 25
lena_filtered_5x5 = cv2.filter2D(lena, -1, h2, borderType=cv2.BORDER_CONSTANT)

plt.figure(figsize=(6,6))
plt.imshow(lena_filtered_5x5, cmap='gray')
plt.title("Ảnh Lena sau khi lọc (5x5 mean filter)")
plt.axis("off")
plt.show()
