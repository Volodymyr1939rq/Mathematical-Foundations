import math
import time
import numpy as np


n = 19
N = 2**n 


dft_mult = 0
dft_add = 0

def calculate_C_k(f, k, N):
    global dft_mult, dft_add
    s = 0j
    for i in range(N):
        arg = -2 * math.pi * k * i / N
        w = complex(math.cos(arg), math.sin(arg))
        s += f[i] * w
        dft_mult += 1
        if i > 0:
            dft_add += 1
    return s / N

def dft(f):
    global dft_mult, dft_add
    dft_mult = dft_add = 0
    N = len(f)
    return np.array([calculate_C_k(f, k, N) for k in range(N)])


fft_mult = 0
fft_add = 0

def fft_sato(f):
    global fft_mult, fft_add
    fft_mult = fft_add = 0
    N_orig = len(f)

    N = 1 << (N_orig - 1).bit_length() if (N_orig & (N_orig - 1)) != 0 else N_orig
    x = [complex(val) for val in (np.pad(f, (0, N - N_orig)) if N != N_orig else f)]

    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j >= bit:
            j -= bit
            bit >>= 1
        j += bit
        if i < j:
            x[i], x[j] = x[j], x[i]

    P = int(math.log2(N))
    for s in range(1, P + 1):
        step = 1 << s
        half_step = step >> 1
        for k in range(half_step):
            arg = -2 * math.pi * k / step
            w = complex(math.cos(arg), math.sin(arg))

            for i in range(k, N, step):
                even = x[i]
                odd = x[i + half_step]

                T = w * odd
                fft_mult += 1

                x[i] = even + T
                x[i + half_step] = even - T
                fft_add += 2

    return np.array(x) / N, N


def fmt(val):
    return f"{int(val):,}".replace(",", " ")

def compare(f_data):
    N_orig = len(f_data)

    t0 = time.perf_counter()
    _, N_fft = fft_sato(f_data)
    t_fft = (time.perf_counter() - t0) * 1000  

    P = math.log2(N_fft)
    m_fft = (N_fft / 2) * P
    a_fft = N_fft * P
    tot_fft = m_fft + a_fft


    if N_orig <= 4096:
        t0 = time.perf_counter()
        dft(f_data)
        t_dft = (time.perf_counter() - t0) * 1000
        mode = "точний замір"
    else:
        N_test = 1024
        t0 = time.perf_counter()
        dft(f_data[:N_test])
        t_test = (time.perf_counter() - t0) * 1000
        t_dft = t_test * (N_orig / N_test) ** 2
        mode = "оцінка O(N^2)"

    m_dft = N_orig ** 2
    a_dft = N_orig * (N_orig - 1)
    tot_dft = m_dft + a_dft

    print(f"\n{'='*72}")
    print(f"Варіант n = {n} | Розмір масиву N = 2^{n} = {fmt(N_fft)}")
    print(f"{'='*72}")
    print(f"{'Показник':<24} | {'ДПФ (' + mode + ')':<21} | {'ШПФ (Юкіо Сато)':<21}")
    print(f"{'-'*72}")
    print(f"{'Час виконання (мс)':<24} | {t_dft:<21.4f} | {t_fft:<21.4f}")
    print(f"{'Кількість множень':<24} | {fmt(m_dft):<21} | {fmt(m_fft):<21}")
    print(f"{'Кількість додавань':<24} | {fmt(a_dft):<21} | {fmt(a_fft):<21}")
    print(f"{'Всього операцій':<24} | {fmt(tot_dft):<21} | {fmt(tot_fft):<21}")
    print(f"{'-'*72}")
    print(f"Виграш ШПФ: у {m_dft / m_fft:.2f} разів менше множень, "
          f"у {a_dft / a_fft:.2f} разів менше додавань.")

f = np.log(np.arange(1, N + 1, dtype=float))

compare(f)