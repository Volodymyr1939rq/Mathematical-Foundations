import numpy as np
import matplotlib.pyplot as plt
import time

n = 19               
N_part1 = 10 + n     
num_part2 = 96 + n   

print(f"Варіант: n = {n}")
print(f"Частина I: N = {N_part1}")
print(f"Частина II: N_число = {num_part2}")

def calculate_C_k(f, k, N):
    
    arg = 2 * np.pi * k * np.arange(N) / N
    Ak = np.sum(f * np.cos(arg)) / N
    Bk = -np.sum(f * np.sin(arg)) / N
    return complex(Ak, Bk)

def dft(f):
    N = len(f)
    return np.array([calculate_C_k(f, k, N) for k in range(N)])

np.random.seed(42)
f_input = np.random.rand(N_part1)


start_time = time.perf_counter()
C_k_array = dft(f_input)
calc_time = time.perf_counter() - start_time


mult_ops = N_part1 * (4 * N_part1 + 2)
add_ops = N_part1 * (2 * (N_part1 - 1))

print("\n--- Частина I: Показники обчислення ---")
print(f"Час обчислення ДПФ (N={N_part1}): {calc_time * 1000:.4f} мс")
print(f"Кількість операцій множення/ділення: {mult_ops}")
print(f"Кількість операцій додавання: {add_ops}")
print(f"Загальна кількість операцій: {mult_ops + add_ops}")

amplitudes = np.abs(C_k_array)
phases = np.angle(C_k_array)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.stem(range(N_part1), amplitudes)
plt.title(f"Спектр амплітуд |C_k| (N={N_part1})")
plt.xlabel("Гармоніка k")
plt.ylabel("|C_k|")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(range(N_part1), phases)
plt.title(f"Спектр фаз arg(C_k) (N={N_part1})")
plt.xlabel("Гармоніка k")
plt.ylabel("arg(C_k) [рад]")
plt.grid(True)
plt.tight_layout()
plt.show()

bin_7bit = [int(b) for b in f"{num_part2:07b}"] 
last_bit = 1 if (n % 2 != 0) else 0             
s_n = np.array(bin_7bit + [last_bit], dtype=float)

print("\n--- Частина II: Вхідні відліки s(nT_delta) ---")
print(f"Число {num_part2} у двійковій системі (7 біт): {bin_7bit}")
print(f"Додано 8-й розряд ({last_bit}) для непарного варіанту {n}")
print(f"Отриманий масив s(n*T_delta): {s_n.astype(int).tolist()}")

# 2. Обчислення комплексних коефіцієнтів ДПФ
N_bits = len(s_n) # 8
C_n_part2 = dft(s_n)

print("\nКомплексні коефіцієнти C_k (k = 0..4):")
for i in range(N_bits // 2 + 1):
    c = C_n_part2[i]
    print(f"C_{i}: Re={c.real:6.3f}, Im={c.imag:6.3f} | |C_{i}|={np.abs(c):.3f}, arg={np.angle(c):.3f} рад")

def s_analog(t_norm, C):
  
    val = np.abs(C[0])
    for k in range(1, 4):
        val += 2 * np.abs(C[k]) * np.cos(2 * np.pi * k * t_norm + np.angle(C[k]))

    val += np.abs(C[4]) * np.cos(2 * np.pi * 4 * t_norm + np.angle(C[4]))
    return val

t_grid = np.linspace(0, 1, 500)
s_cont = s_analog(t_grid, C_n_part2)

plt.figure(figsize=(10, 4))
plt.plot(t_grid, s_cont, 'b-', label="Відтворений аналоговий сигнал s(t)")
plt.stem(np.arange(8) / 8.0, s_n, linefmt='r--', markerfmt='ro', basefmt=" ", label="Дискретні відліки s[n]")
plt.title(f"Відтворення сигналу для послідовності {''.join(map(str, s_n.astype(int)))}")
plt.xlabel("Нормований час t / T_c")
plt.ylabel("s(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- Частина III: Результати ЗДПФ ---")
def idft(C):
    N = len(C)
    res = np.zeros(N)
    for n_idx in range(N):
     
        val = sum(C[k] * np.exp(1j * 2 * np.pi * k * n_idx / N) for k in range(N))
        res[n_idx] = val.real
    return res

s_reconstructed_samples = idft(C_n_part2)
print("Відтворені відліки s(nT_delta):", np.round(s_reconstructed_samples, 3))
print("Початкові відліки:            ", s_n)

print(f"\nАналітичний розрахунок n=0: s(0) = {s_reconstructed_samples[0]:.3f} (Очікувалось: {s_n[0]})")
print(f"Аналітичний розрахунок n=1: s(1) = {s_reconstructed_samples[1]:.3f} (Очікувалось: {s_n[1]})")