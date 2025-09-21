import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt

# ============================
# (a) Tạo tín hiệu giả lập "handel"
# ============================
Fs = 8192  # tần số lấy mẫu (Hz)
duration = 5  # 5 giây
t = np.linspace(0, duration, Fs*duration, endpoint=False)

# tạo âm thanh bằng cách cộng nhiều sóng sin (440Hz, 660Hz, 880Hz)
x = 0.5*np.sin(2*np.pi*440*t) + 0.3*np.sin(2*np.pi*660*t) + 0.2*np.sin(2*np.pi*880*t)

print(f"(a) Playing original sound, Fs = {Fs}")
sd.play(x, Fs)
sd.wait()

# ============================
# (b) Lấy cách 2 mẫu -> Fs/2
# ============================
x2 = x[::2]
Fs2 = Fs // 2
print(f"(b) Playing downsampled sound, Fs = {Fs2}")
sd.play(x2, Fs2)
sd.wait()

# ============================
# (c) Lấy cách 4 mẫu -> Fs/4
# ============================
x4 = x[::4]
Fs4 = Fs // 4
print(f"(c) Playing downsampled sound, Fs = {Fs4}")
sd.play(x4, Fs4)
sd.wait()

# ============================
# (d) Lưu file WAV
# ============================
wavfile.write("handel_down4.wav", Fs4, x4.astype(np.float32))
print("(d) Saved file: handel_down4.wav")

# ============================
# (bonus) Vẽ dạng sóng để so sánh
# ============================
plt.figure(figsize=(12, 6))
plt.subplot(3,1,1)
plt.plot(x[:400])
plt.title("Original Fs=8192")

plt.subplot(3,1,2)
plt.plot(x2[:200])
plt.title("Downsample Fs=4096")

plt.subplot(3,1,3)
plt.plot(x4[:100])
plt.title("Downsample Fs=2048")

plt.tight_layout()
plt.show()
