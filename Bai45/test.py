import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# đường dẫn tới ảnh lena (thay bằng đường dẫn của bạn nếu cần)
img_path = r"F:\XLTHS\Homework\Bai15\lena.jpg"  # <-- sửa lại nếu cần

if not os.path.isfile(img_path):
    raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

# (a) Load và chuyển sang grayscale
lena = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if lena is None:
    raise RuntimeError("Không thể đọc ảnh. Kiểm tra định dạng/đường dẫn.")

# chuyển về float để tính toán (tránh tràn số)
lena_f = lena.astype(np.float32)

# (b) Kernel 3x3 (Sobel-like) -- phát hiện biên theo chiều ngang/vertical edges
h_b = np.array([[1,  0, -1],
                [2,  0, -2],
                [1,  0, -1]], dtype=np.float32)

# (c) Kernel 3x3 khác (Sobel-like) -- phát hiện biên theo chiều dọc/horizontal edges
h_c = np.array([[ 1,  2,  1],
                [ 0,  0,  0],
                [-1, -2, -1]], dtype=np.float32)

# Áp dụng lọc với zero boundary (BORDER_CONSTANT)
# sử dụng ddepth = cv2.CV_32F để giữ dấu và độ chính xác
res_b = cv2.filter2D(lena_f, ddepth=cv2.CV_32F, kernel=h_b, borderType=cv2.BORDER_CONSTANT)
res_c = cv2.filter2D(lena_f, ddepth=cv2.CV_32F, kernel=h_c, borderType=cv2.BORDER_CONSTANT)

# Để hiển thị: ta thường lấy giá trị tuyệt đối (magnitude) rồi chuẩn hoá về 0-255
def to_display(img_float):
    absimg = np.abs(img_float)
    maxv = absimg.max() if absimg.max() != 0 else 1.0
    disp = (absimg / maxv * 255.0).astype(np.uint8)
    return disp

disp_b = to_display(res_b)
disp_c = to_display(res_c)

# Hiển thị ảnh gốc và 2 kết quả
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(lena, cmap='gray')
plt.title("Lena gốc")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(disp_b, cmap='gray')
plt.title("Kết quả (b) kernel [[1 0 -1],[2 0 -2],[1 0 -1]]")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(disp_c, cmap='gray')
plt.title("Kết quả (c) kernel [[1 2 1],[0 0 0],[-1 -2 -1]]")
plt.axis('off')

plt.tight_layout()
plt.show()

# Nếu muốn lưu kết quả
out_dir = r"F:\XLTHS\Homework\Bai15\results"
os.makedirs(out_dir, exist_ok=True)
cv2.imwrite(os.path.join(out_dir, "lena_edge_b.png"), disp_b)
cv2.imwrite(os.path.join(out_dir, "lena_edge_c.png"), disp_c)
print(f"Đã lưu kết quả vào: {out_dir}")
