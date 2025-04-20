def butter_low_filter(data, fs, cut_hz, order):

    from scipy.signal import butter, filtfilt

    C = 0.802
    b, a = butter(order, (cut_hz / (fs / 2) / C), btype="low")
    out = filtfilt(b, a, data)
    return out
