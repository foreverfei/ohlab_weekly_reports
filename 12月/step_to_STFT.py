from scipy.signal import stft
from matplotlib import pyplot as plt
import numpy as np
'''
STFT:
    scipy.signal.stft(x, fs, window, nperseg, noverlap, nfft, detrend, return_oneside, boundary, padded, axis)

    1.x ：传入STFT变换的时域信号
    2.fs : 时域信号的采样频率，默认为1.0
    3.window : 时域信号分割的时候需要的窗函数，常用的窗函数有boxcar，triang，hamming， hann等
    4.nperseg : 窗函数的长度，默认值为256
    5.noverlap : 窗函数重叠数，默认为窗长的一半
    6.nfft : FFT的长度，默认为nperseg，如果设置为大于nperseg会自动进行0填充
    7.return_oneside : True返回复数实部，None返回复数，默认为False
'''
'''
STFT 返回值:
    f: 频率数组（Frequency bins）
    
    包含频谱中的频率值，单位为赫兹（Hz）。
    t: 时间数组（Time bins）
    
    包含时间轴上的时刻，表示每个窗口中心的时间。
    Zxx: STFT 结果（STFT of the signal）
    
    是一个复数数组，表示每个时间和频率组合的幅度和相位。
    其形状是 (n_fft / 2 + 1, n_windows)，其中 n_fft 是FFT的长度，n_windows 是窗口的数量。
'''
if __name__ == '__main__':

    fs = 1024 # 采样频率
    x = np.linspace(0, 1, fs , endpoint=False)
    y = []
    for i in range(3) :
        f = np.random.randint(180,361)
        y.append(np.sin(2 * np.pi * f* x))
        print(f)
    final_y = np.concatenate(y)

    frequencies , t , spectrum = stft(final_y, fs, nperseg=312)

    # print(frequencies)
    # print(frequencies.shape)
    # print(t)
    # print(t.shape)
    # print(spectrum)
    # print(spectrum.shape)

    # 绘制时频图
    plt.pcolormesh(t, frequencies, np.abs(spectrum), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()