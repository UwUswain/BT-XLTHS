import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# (a) Tạo tín hiệu giả lập giống "handel"
Fs = 8192  # tần số lấy mẫu
t = np.linspace(0, 5, Fs*5, endpoint=False)  # 5 giây

# tạo âm thanh bằng cách cộng nhiều sin khác nhau
x = 0.5*np.sin(2*np.pi*440*t) + 0.3*np.sin(2*np.pi*660*t) + 0.2*np.sin(2*np.pi*880*t)

print(f"(a) Playing original fake handel, Fs={Fs}")
sd.play(x, Fs)
sd.wait()

# (b) Chọn mỗi 2 mẫu
x2 = x[::2]
Fs2 = Fs // 2
print(f"(b) Playing downsampled sound (Fs/2 = {Fs2})")
sd.play(x2, Fs2)
sd.wait()

# (c) Chọn mỗi 4 mẫu
x4 = x[::4]
Fs4 = Fs // 4
print(f"(c) Playing downsampled sound (Fs/4 = {Fs4})")
sd.play(x4, Fs4)
sd.wait()

# (d) Lưu ra file WAV
wavfile.write("handel_fake_down4.wav", Fs4, x4.astype(np.float32))
print("(d) Saved: handel_fake_down4.wav")
