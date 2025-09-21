# laplacian_image_processing.py
import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# (1) Đọc ảnh Lena (bạn cần file 'lena.png' trong cùng thư mục)
# nếu không có sẵn, có thể tải từ internet hoặc thay ảnh bất kỳ
img = cv2.imread("F:\XLTHS\Homework\Bai54\lena.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Không tìm thấy file lena.png, hãy để ảnh cùng thư mục.")

# (2) Định nghĩa bộ lọc Laplacian
h = np.array([[0, 1, 0],
              [1, -4, 1],
              [0, 1, 0]])

# (a) Ảnh sau khi lọc với Laplacian
laplace_img = convolve2d(img, h, mode='same', boundary='symm')

# Chuẩn hóa về 0..255 để hiển thị
laplace_img_disp = cv2.normalize(laplace_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# (b) Ảnh tăng cường biên = ảnh gốc - Laplacian
edge_enhanced = img.astype(np.float32) - laplace_img
edge_enhanced_disp = cv2.normalize(edge_enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# (c) Hiển thị kết quả
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Ảnh gốc (Lena)")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(laplace_img_disp, cmap='gray')
plt.title("Kết quả (a): Laplacian")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(edge_enhanced_disp, cmap='gray')
plt.title("Kết quả (b),(c): Ảnh tăng cường biên")
plt.axis("off")

plt.show()
