# problem50_reverb.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import os

# ======================
# (b) Implementation
# ======================

def reverb_old(x, a, D):
    """
    y[n] = x[n] + a*y[n-D]
    Recursive (feedback)
    """
    N = len(x)
    y = np.zeros(N, dtype=np.float32)
    for n in range(N):
        y[n] = x[n]
        if n - D >= 0:
            y[n] += a * y[n - D]
    return y

def reverb_new(x, a, D):
    """
    y[n] = a*y[n-D] + x[n-D]
    FIR (feedforward only)
    """
    N = len(x)
    y = np.zeros(N, dtype=np.float32)
    for n in range(N):
        if n - D >= 0:
            y[n] = a * y[n - D] + x[n - D]
        else:
            y[n] = 0
    return y

def normalize_audio(x, peak=0.98):
    maxv = np.max(np.abs(x))
    if maxv == 0:
        return x
    return x * (peak / maxv)

# ======================
# MAIN PROGRAM
# ======================
if __name__ == "__main__":
    Fs = 8192
    a = 0.7
    tau = 0.05   # 50 ms
    D = int(round(tau * Fs))
    print(f"Fs={Fs}, a={a}, tau={tau}s, D={D} samples")

    # (c) Impulse response
    L = (10+2)*D
    x_imp = np.zeros(L, dtype=np.float32)
    x_imp[0] = 1.0

    h_old = reverb_old(x_imp, a, D)
    h_new = reverb_new(x_imp, a, D)

    t = np.arange(L)/Fs
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.stem(t, h_old, basefmt=" ")
    plt.xlim(0, (10+1)*D/Fs)
    plt.title("Impulse Response - Old System: y[n]=x[n]+a y[n-D]")
    plt.xlabel("Time (s)"); plt.ylabel("h_old[n]")

    plt.subplot(2,1,2)
    plt.stem(t, h_new, basefmt=" ")
    plt.xlim(0, (10+1)*D/Fs)
    plt.title("Impulse Response - New System: y[n]=a y[n-D] + x[n-D]")
    plt.xlabel("Time (s)"); plt.ylabel("h_new[n]")
    plt.tight_layout()
    plt.show()

    # (d) Apply to signal
    if os.path.isfile("handel.wav"):
        Fs, x = wavfile.read("handel.wav")
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        elif x.dtype == np.int32:
            x = x.astype(np.float32) / 2147483648.0
        elif x.dtype == np.uint8:
            x = (x.astype(np.float32) - 128) / 128.0
        else:
            x = x.astype(np.float32)
        if x.ndim == 2:  # stereo -> mono
            x = x.mean(axis=1)
    else:
        print("[info] handel.wav not found, creating test signal (sine mix).")
        duration = 4.0
        t = np.linspace(0, duration, int(Fs*duration), endpoint=False)
        x = 0.5*np.sin(2*np.pi*440*t) + 0.3*np.sin(2*np.pi*660*t) + 0.2*np.sin(2*np.pi*880*t)
        x = x.astype(np.float32)

    y_old = normalize_audio(reverb_old(x, a, D))
    y_new = normalize_audio(reverb_new(x, a, D))

    print("Playing old reverb...")
    sd.play(y_old, Fs); sd.wait()
    print("Playing new reverb...")
    sd.play(y_new, Fs); sd.wait()

    # Save to file
    wavfile.write("reverb_old.wav", Fs, (y_old*32767).astype(np.int16))
    wavfile.write("reverb_new.wav", Fs, (y_new*32767).astype(np.int16))
    print("Saved: reverb_old.wav, reverb_new.wav")
