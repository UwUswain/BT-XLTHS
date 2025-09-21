import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# =========================
# (1) Đọc ảnh Lena (grayscale)
# =========================
img = cv2.imread("F:\XLTHS\Homework\Bai53\lena.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Không tìm thấy file 'lena.png'. Đặt ảnh cùng thư mục với script.")

# =========================
# (2) Định nghĩa impulse response
# h[-1]=1, h[0]=-2, h[1]=1
# =========================
h = np.array([1, -2, 1], dtype=np.float32)

# =========================
# (3) Lọc row-by-row
# kernel ngang 1x3
# =========================
h_row = h.reshape(1, 3)
img_row_filtered = convolve2d(img, h_row, mode='same', boundary='symm')
img_row_disp = cv2.normalize(img_row_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# =========================
# (4) Lọc column-by-column
# kernel dọc 3x1
# =========================
h_col = h.reshape(3, 1)
img_col_filtered = convolve2d(img, h_col, mode='same', boundary='symm')
img_col_disp = cv2.normalize(img_col_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# =========================
# (5) Kết hợp row + column (Laplacian 2D approximation)
# =========================
img_combined = img_row_filtered + img_col_filtered
img_combined_disp = cv2.normalize(img_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# =========================
# (6) Hiển thị kết quả
# =========================
plt.figure(figsize=(15,6))

plt.subplot(1,4,1)
plt.imshow(img, cmap='gray')
plt.title("Ảnh gốc")
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(img_row_disp, cmap='gray')
plt.title("Row-by-row")
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(img_col_disp, cmap='gray')
plt.title("Column-by-column")
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(img_combined_disp, cmap='gray')
plt.title("Row+Column (All edges)")
plt.axis('off')

plt.show()
