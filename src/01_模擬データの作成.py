# %% == 模擬データの作成 ================ 
import numpy as np
from scipy import signal
import importlib
import utils
importlib.reload(utils)
# %% == ホワイトノイズの作成 ============
FS = 48000
duration = 3
samples = int(FS * duration)
wn = np.random.normal(0, 1, samples)

utils.plot_spectrogram(wn, FS)

# %% == ピークフィルタをかけてみる ======= 
f0 = 1000
Q = 100
b,a = signal.iirpeak(f0, Q, fs=FS)
wn_1 = signal.lfilter(b, a, wn)

f2 = 4000
Q = 100
b,a = signal.iirpeak(f2, Q, fs=FS)
wn_2 = signal.lfilter(b, a, wn)

def rms(x):
    return np.sqrt(np.mean(x**2))

wn_all = wn_1 + wn_2
wn_all *= rms(wn) / rms(wn_all)

utils.plot_spectrogram(wn_all, FS)

# 🤔ノッチができてしまいよくないかもしれない。
# %% == フィルタを直列結合にする（上記は並列）
b1, a1 = signal.iirpeak(1000, 400, fs=FS)
b2, a2 = signal.iirpeak(5000, 400, fs=FS)

sos1 = signal.tf2sos(b1, a1)
sos2 = signal.tf2sos(b2, a2)
sos = np.vstack([sos1, sos2])  # 直列に積む

wn_all = signal.sosfilt(sos, wn)
utils.plot_spectrogram(wn_all, FS)

# 🤔他方法も試してみる。
# 🤔削れ過ぎてもノイズではなくなるからよくはないよな。

# %% == FIRの方法も試してみる =============
numtaps = 256

bp1 = signal.firwin(numtaps, [950, 1050], pass_zero=False, fs=FS)
bp2 = signal.firwin(numtaps, [3950, 4050], pass_zero=False, fs=FS)
fir = bp1 + bp2
wn_filt = signal.lfilter(fir, 1.0, wn)
utils.plot_spectrogram(wn_filt, FS)

# 🤔FIRが悪くなさそうなので、FIRで行く。

# %% == FIRで中心周波数が動くフィルタを試す。=
frame_len = int(0.05 * FS)   
hop = frame_len // 2      
window = np.hanning(frame_len)

n_frames = (samples - frame_len) // hop

numtaps = 1024

# 周波数トラジェクトリ
f1_traj = np.linspace(1000, 4000, n_frames)
f2_traj = np.linspace(4000, 10000, n_frames)

wn_dynamic = np.zeros(samples)

for i in range(n_frames):
    start = i * hop
    end = start + frame_len
    x = wn[start:end] * window

    # --- フィルタ1 ---
    bw1 = 200
    bp1 = signal.firwin(numtaps,
                        [f1_traj[i]-bw1/2, f1_traj[i]+bw1/2],
                        pass_zero=False, fs=FS)

    # --- フィルタ2 ---
    bw2 = 500
    bp2 = signal.firwin(numtaps,
                        [f2_traj[i]-bw2/2, f2_traj[i]+bw2/2],
                        pass_zero=False, fs=FS)

    fir = bp1 + bp2
    y = signal.lfilter(fir, 1.0, x)

    # Overlap-Add
    wn_dynamic[start:end] += y[:frame_len] * window

utils.plot_spectrogram(wn_dynamic, FS)

# 🤔悪くない

# %% == bwも制御してみる ==================
frame_len = int(0.05 * FS)
hop = frame_len // 2
window = np.hanning(frame_len)

n_frames = (samples - frame_len) // hop

numtaps = 1024
nyq = FS / 2.0

# 周波数トラジェクトリ
f1_traj = np.linspace(1000, 4000, n_frames)   # 1k -> 4k
f2_traj = np.linspace(4000, 10000, n_frames)  # 4k -> 10k

# bandwidthトラジェクトリ
bw1_traj = np.linspace(150.0, 1000.0, n_frames)   # 例: 150Hz -> 800Hz
bw2_traj = np.linspace(1000.0, 100.0, n_frames)   # 例: 600Hz -> 200Hz（逆に狭める）

def band_edges(fc, bw, margin=10.0):
    """[lo, hi] を安全に切り出し（最小限の簡易クランプ）"""
    bw = max(float(bw), 1.0)
    lo = max(margin, fc - bw/2.0)
    hi = min(nyq - margin, fc + bw/2.0)
    if hi <= lo:
        lo = max(margin, min(fc - 0.5*margin, nyq - margin*1.5))
        hi = lo + margin
    return lo, hi
# -----------------------------

wn_dynamic = np.zeros(samples)

for i in range(n_frames):
    start = i * hop
    end = start + frame_len
    x = wn[start:end] * window

    # --- フィルタ1（bwをフレームごとに変更） ---
    bw1 = bw1_traj[i]
    lo1, hi1 = band_edges(f1_traj[i], bw1)
    bp1 = signal.firwin(numtaps, [lo1, hi1], pass_zero=False, fs=FS)

    # --- フィルタ2（bwをフレームごとに変更） ---
    bw2 = bw2_traj[i]
    lo2, hi2 = band_edges(f2_traj[i], bw2)
    bp2 = signal.firwin(numtaps, [lo2, hi2], pass_zero=False, fs=FS)

    # 合成して適用
    fir = bp1 + bp2
    y = signal.lfilter(fir, 1.0, x)

    # Overlap-Add
    wn_dynamic[start:end] += y[:frame_len] * window

utils.plot_spectrogram(wn_dynamic, FS)

# %%
