import math
import time

import numpy as np
import matplotlib.pyplot as plt


n = 19                 
N = 10 + n             
num = 96 + n         
EPS = 1e-12            

print(f"Варіант: n = {n}")
print(f"Частина I: N = {N}")
print(f"Частина II: число = {num}")

mult_ops = 0
add_ops = 0

def fourier_term(A_k, B_k, k, N, i):
   
    arg = 2 * math.pi * k * i / N
    return A_k * math.cos(arg) - B_k * math.sin(arg)


def calculate_C_k(f, k, N):
  
    global mult_ops, add_ops
    sA = 0.0
    sB = 0.0
    for i in range(N):
        arg = 2 * math.pi * k * i / N   
        sA += f[i] * math.cos(arg)        
        sB += f[i] * math.sin(arg)       
        mult_ops += 2
        if i >0:
         add_ops += 2
                         
    return complex(sA / N, -sB / N)


def dft(f):
    N = len(f)
    return np.array([calculate_C_k(f, k, N) for k in range(N)])


def phase(c):
    return 0.0 if abs(c) < EPS else math.atan2(c.imag, c.real)


np.random.seed(42)
f_input = np.random.rand(N)

mult_ops = add_ops = 0
start_time = time.perf_counter()
C_k_array = dft(f_input)
calc_time = time.perf_counter() - start_time

print("\n--- Частина I: Показники обчислення ---")
print(f"Час обчислення ДПФ (N={N}): {calc_time * 1000:.4f} мс")
print(f"Кількість множень/ділень: {mult_ops}  ")
print(f"Кількість додавань:       {add_ops}  ")
print(f"Загальна кількість операцій: {mult_ops + add_ops}")

i_test = 3
rec = sum(fourier_term(C_k_array[k].real, C_k_array[k].imag, k, N, i_test) for k in range(N))
print(f"Перевірка fourier_term: f[{i_test}] = {f_input[i_test]:.6f}, "
      f"сума членів ряду = {rec:.6f}")

amplitudes = np.abs(C_k_array)
phases = np.array([phase(c) for c in C_k_array])

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.stem(range(N), amplitudes)
plt.title(f"Спектр амплітуд |C_k| (N={N})")
plt.xlabel("Гармоніка k")
plt.ylabel("|C_k|")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(range(N), phases)
plt.title(f"Спектр фаз arg(C_k) (N={N})")
plt.xlabel("Гармоніка k")
plt.ylabel("arg(C_k) [рад]")
plt.grid(True)
plt.tight_layout()
plt.savefig("lab2_part1_spectra.png", dpi=150)
plt.show()


bin_7bit = [int(b) for b in f"{num:07b}"]
last_bit = 1 if (n % 2 != 0) else 0                  
s_n = np.array([last_bit] + bin_7bit, dtype=float)   
N8 = len(s_n)                                        
code = "".join(str(int(b)) for b in s_n)

print("\n--- Частина II: Вхідні відліки s(nT_delta) ---")
print(f"Число {num} у двійковій системі (7 біт): {''.join(map(str, bin_7bit))}")
print(f"Отриманий масив s(n*T_delta): {s_n.astype(int).tolist()}  ->  {code}")

C_n = dft(s_n)

print("\nКомплексні коефіцієнти C_k (k = 0..4):")
for k in range(N8 // 2 + 1):
    c = C_n[k]
    if abs(c) < EPS:
        arg_txt = "не визначений"
    else:
        arg_txt = f"{phase(c):+.4f} рад ({math.degrees(phase(c)):+.1f} град)"
    print(f"C_{k} = {c.real:+.4f} {c.imag:+.4f}j | |C_{k}| = {abs(c):.4f}, arg C_{k} = {arg_txt}")


def s_analog(t_norm, C):
    
    Nc = len(C)
    val = C[0].real
    for k in range(1, Nc // 2):
        val += 2 * abs(C[k]) * np.cos(2 * np.pi * k * t_norm + phase(C[k]))
    val += abs(C[Nc // 2]) * np.cos(np.pi * Nc * t_norm + phase(C[Nc // 2]))
    return val

expr = f"s(t) = {C_n[0].real:.3f}"
for k in range(1, N8 // 2):
    expr += f" + {2 * abs(C_n[k]):.3f}*cos({2 * k}*pi*t/Tc {phase(C_n[k]):+.3f})"
expr += f" + {abs(C_n[N8 // 2]):.3f}*cos({N8}*pi*t/Tc {phase(C_n[N8 // 2]):+.3f})"
print("\nАналітичний вираз відтвореного сигналу:")
print(expr)

t_grid = np.linspace(0, 1, 500)
s_cont = s_analog(t_grid, C_n)

plt.figure(figsize=(10, 4))
plt.plot(t_grid, s_cont, "b-", label="Відтворений аналоговий сигнал s(t)")
plt.stem(np.arange(N8) / N8, s_n, linefmt="r--", markerfmt="ro", basefmt=" ",
         label="Дискретні відліки s[n]")
plt.title(f"Відтворення сигналу для послідовності {code}")
plt.xlabel("Нормований час t / T_c")
plt.ylabel("s(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lab2_part2_reconstruction.png", dpi=150)
plt.show()


def idft(C):
   
    Nc = len(C)
    res = np.zeros(Nc, dtype=complex)
    for m in range(Nc):
        res[m] = sum(C[k] * np.exp(1j * 2 * np.pi * k * m / Nc) for k in range(Nc))
    return res


print("\n--- Частина III: Результати ОДПФ ---")
s_back = idft(C_n)
for m in range(N8):
    print(f"s({m}T_delta) = {s_back[m].real:+.4f}   (Im = {s_back[m].imag:+.1e}),  "
          f"вихідне = {int(s_n[m])}")
print(f"Макс. відхилення від вихідних відліків: {np.max(np.abs(s_back - s_n)):.2e}")

print("\nАналітичний вираз s(0):")
print("s(0) = C0 + 2Re C1 + 2Re C2 + 2Re C3 + Re C4")
vals0 = [C_n[0].real] + [2 * C_n[k].real for k in (1, 2, 3)] + [C_n[4].real]
print("     = " + " ".join(f"{v:+.3f}" for v in vals0) + f" = {sum(vals0):.3f}")

print("\nАналітичний вираз s(T_delta):")
print("s(T_delta) = C0 + 2|C1|cos(pi/4 + phi1) + 2|C2|cos(pi/2 + phi2)"
      " + 2|C3|cos(3pi/4 + phi3) + |C4|cos(pi + phi4)")
vals1 = [C_n[0].real]
for k in (1, 2, 3):
    vals1.append(2 * abs(C_n[k]) * math.cos(math.pi * k / 4 + phase(C_n[k])))
vals1.append(abs(C_n[4]) * math.cos(math.pi + phase(C_n[4])))
print("           = " + " ".join(f"{v:+.3f}" for v in vals1) + f" = {sum(vals1):.3f}")