from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio

def plot_spectrogram(data, fs, **kwargs):

    # デフォルトパラメータ
    nperseg = kwargs.pop("nperseg", 1024)
    noverlap = kwargs.pop("noverlap", int(nperseg * 0.75))

    # STFT
    f, t, Zxx = signal.stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap, **kwargs)

    # 振幅スペクトル → dB
    Z = np.abs(Zxx)
    Z_db = 10 * np.log10(Z / np.max(Z))

    # 描画
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.pcolormesh(t, f, Z_db, cmap="turbo",vmin=-80, vmax=0)
    cbar = fig.colorbar(img, ax=ax)
    ax.set(xlabel="Time [s]", ylabel="Frequency [Hz]")
    ax.set(ylim=[20, 10000], yscale="linear")
    cbar.set_label("Level [dB]")
    
    # 音声再生
    return Audio(data, rate=fs)
    

