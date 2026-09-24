import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

def get_variant_params(n):
   
    if n % 2 == 0:
        print(f"Варіант n={n} (Парний). Інтервал [-π, π]. Функція експоненціальна.")
        a, b = -np.pi, np.pi
        def exact_f(x):
            return x * n * np.exp(-(x**2) / n)
        is_even = True
    else:
        print(f"Варіант n={n} (Непарний). Інтервал [0, π]. Функція тригонометрична.")
        a, b = 0, np.pi
        def exact_f(x):
            return n * np.sin(np.pi * n * x)
        is_even = False
        
    T = b - a
    omega = 2 * np.pi / T
    return a, b, T, omega, exact_f, is_even

def fourier_approx(x, a_coeffs, b_coeffs, N, omega):
    result = a_coeffs[0] / 2
    for k in range(1, N + 1):
        result += a_coeffs[k] * np.cos(k * omega * x) + b_coeffs[k] * np.sin(k * omega * x)
    return result

def calc_relative_error(y_exact, y_approx):
    error_norm = np.linalg.norm(y_exact - y_approx)
    exact_norm = np.linalg.norm(y_exact)
    if exact_norm == 0:
        return 0.0
    return (error_norm / exact_norm) * 100

def save_to_file(filename, n, N, a, b, error):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Варіант: {n} | Порядок наближення N: {N}\n")
        f.write("-" * 35 + "\n")
        f.write("Коефіцієнти тригонометричного ряду Фур'є:\n")
        f.write(f"k= 0 | a_0 = {a[0]:.6f}\n")
        for k in range(1, N + 1):
            f.write(f"k={k:2d} | a_{k} = {a[k]:.6f} | b_{k} = {b[k]:.6f}\n")
        f.write("-" * 35 + "\n")
        f.write(f"Відносна похибка наближення: {error:.4f}%\n")

def plot_all_results(x_vals, y_exact, y_approx, a_coeffs, b_coeffs, N, omega, n):
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Аналіз ряду Фур'є для варіанту n={n} (N={N})", fontsize=16)

    axs[0, 0].plot(x_vals, y_exact, label="Оригінальна функція f(x)", color="blue", alpha=0.7)
    axs[0, 0].plot(x_vals, y_approx, label=f"Наближення (N={N})", color="red", linestyle="--")
    axs[0, 0].set_title("Функція та її наближення")
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("f(x)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].set_title("Окремі гармоніки (перші 5)")
    axs[0, 1].plot(x_vals, np.full_like(x_vals, a_coeffs[0] / 2), label="a_0 / 2 (Постійна)", color='black', linestyle=':')
    for k in range(1, min(N + 1, 6)):
        harmonic = a_coeffs[k] * np.cos(k * omega * x_vals) + b_coeffs[k] * np.sin(k * omega * x_vals)
        axs[0, 1].plot(x_vals, harmonic, label=f"Гармоніка k={k}")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("Амплітуда")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    k_vals = np.arange(0, N + 1)

    axs[1, 0].stem(k_vals, a_coeffs, basefmt='black')
    axs[1, 0].set_title("Коефіцієнти $a_k$ (Частотна область)")
    axs[1, 0].set_xlabel("k")
    axs[1, 0].set_ylabel("$a_k$")
    axs[1, 0].grid(True)

    axs[1, 1].stem(k_vals, b_coeffs, basefmt='black')
    axs[1, 1].set_title("Коефіцієнти $b_k$ (Частотна область)")
    axs[1, 1].set_xlabel("k")
    axs[1, 1].set_ylabel("$b_k$")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

def main():
    try:
        n = int(input("Введіть номер студента у журналі (n): "))
    except ValueError:
        n = 19
        print("Помилка вводу. Використовується варіант 19.")

    N = 10
    
    a_limit, b_limit, T, omega, exact_f, is_even = get_variant_params(n)
    
    print(f"Обчислення коефіцієнтів для N={N}...")
    
    a_coeffs = [0.0] * (N + 1)
    b_coeffs = [0.0] * (N + 1)

    if is_even:
        def integrand_a0(x):
            return exact_f(x)
        integral_a0, _ = spi.quad(integrand_a0, a_limit, b_limit, limit=1000)
        a_coeffs[0] = np.round((2 / T) * integral_a0, 6)

        for k in range(1, N + 1):
            def integrand_ak(x):
                return exact_f(x) * np.cos(k * omega * x)
            integral_ak, _ = spi.quad(integrand_ak, a_limit, b_limit, limit=1000)
            a_coeffs[k] = (2 / T) * integral_ak
    else:
        for k in range(1, N + 1):
            def integrand_bk(x):
                return exact_f(x) * np.sin(k * omega * x)
            integral_bk, _ = spi.quad(integrand_bk, a_limit, b_limit, limit=1000)
            b_coeffs[k] = (2 / T) * integral_bk

    x_vals = np.linspace(a_limit, b_limit, 1000)
    y_exact = exact_f(x_vals)
    y_approx = np.array([fourier_approx(x, a_coeffs, b_coeffs, N, omega) for x in x_vals])
    
    error = calc_relative_error(y_exact, y_approx)
    save_to_file('fourier_results.txt', n, N, a_coeffs, b_coeffs, error)
    
    print(f"Готово! Відносна похибка: {error:.2f}%")
    print("Результати збережено у файл 'fourier_results.txt'")

    plot_all_results(x_vals, y_exact, y_approx, a_coeffs, b_coeffs, N, omega, n)

if __name__ == "__main__":
    main()