# reverb_task.py
import os
import numpy as np
from scipy.io import wavfile
import sounddevice as sd

def load_or_create_signal(path="handel.wav", Fs_default=8192, duration=5.0):
    """
    Nếu path tồn tại và đọc được WAV -> trả về (Fs, x_float)
    Ngược lại tạo tín hiệu giả (multi-sine) với Fs_default và duration (s).
    x_float là float32 trong khoảng [-1, 1].
    """
    if os.path.isfile(path):
        Fs, x = wavfile.read(path)
        # Convert to float in [-1,1] depending on dtype
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        elif x.dtype == np.int32:
            x = x.astype(np.float32) / 2147483648.0
        elif x.dtype == np.uint8:
            x = (x.astype(np.float32) - 128) / 128.0
        else:
            x = x.astype(np.float32)
        # if stereo, convert to mono by averaging channels
        if x.ndim == 2:
            x = x.mean(axis=1)
        return Fs, x
    else:
        print(f"[info] '{path}' not found — tạo tín hiệu giả thay thế (multi-sine).")
        Fs = Fs_default
        t = np.linspace(0, duration, int(Fs*duration), endpoint=False)
        x = 0.5*np.sin(2*np.pi*440*t) + 0.3*np.sin(2*np.pi*660*t) + 0.2*np.sin(2*np.pi*880*t)
        # đảm bảo nằm trong [-1,1]
        x = x.astype(np.float32)
        x /= np.max(np.abs(x)) + 1e-12
        return Fs, x

def recursive_reverb(x, a, D):
    """
    Áp dụng y[n] = x[n] + a * y[n-D], với y initial = zeros.
    Trả về y (float32).
    Thực hiện theo chỉ số để đảm bảo thứ tự tính toán.
    """
    N = len(x)
    y = np.zeros(N, dtype=np.float32)
    for n in range(N):
        yn = x[n]
        if n - D >= 0:
            yn += a * y[n - D]
        y[n] = yn
    return y

def normalize_audio(x, peak=0.98):
    maxv = np.max(np.abs(x))
    if maxv == 0:
        return x
    scale = peak / maxv
    return x * scale

def save_wav(filename, Fs, x_float):
    # convert to int16
    x_int16 = np.int16(np.clip(x_float, -1.0, 1.0) * 32767)
    wavfile.write(filename, Fs, x_int16)
    print(f"[saved] {filename}")

def main():
    Fs, x = load_or_create_signal(path="handel.wav", Fs_default=8192, duration=8.0)
    print(f"Sampling rate Fs = {Fs} Hz, signal length = {len(x)} samples ({len(x)/Fs:.2f} s)")

    a = 0.7
    taus = [0.050, 0.100, 0.500]  # seconds: 50ms, 100ms, 500ms

    for tau in taus:
        D = int(round(tau * Fs))
        print(f"\n-- tau = {tau*1000:.0f} ms -> D = {D} samples, a = {a}")
        y = recursive_reverb(x, a, D)

        # Normalize to avoid clipping on playback
        y_norm = normalize_audio(y, peak=0.98)

        # Play (may be long)
        print("Playing...")
        sd.play(y_norm, Fs)
        sd.wait()

        # Save file
        outname = f"handel_reverb_tau{int(tau*1000)}ms_a{int(a*100)}.wav"
        save_wav(outname, Fs, y_norm)

    print("\nDone. Files saved for each tau. Listen and compare.")
    print("Notes: this implementation is a single-feedback comb filter (one delayed feedback).")
    print("Short tau -> dense, maybe more 'room' feeling; long tau -> distinct echoes.")

if __name__ == "__main__":
    main()
