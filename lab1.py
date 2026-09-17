import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

n=19
interval=np.pi
omega=2*np.pi/interval

def exact_f(x):
    return n*np.sin(np.pi*n*x)

def calc_coefficient(k):
    integrand=lambda x:exact_f(x)*np.cos(k*omega*x)
    integral,_=spi.quad(integrand,0,interval,limit=1000)
    return (2/interval)*integral

def calc_coefficient_bk(k):
    if k==0:
        return 0.0

    integrand=lambda x:exact_f(x)*np.sin(k*omega*x)
    integral,_=spi.quad(integrand,0,interval,limit=1000)
    return (2/interval)*integral

def fourier_approx(x,a,b,N):
    result=a[0]/2
    for k in range(1,N+1):
        result+=a[k]*np.cos(k*omega*x)+b[k]*np.sin(k*omega*x)
    return result

def plot_garmonics(a,b,N):
    k_vals=np.arange(0,N+1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.stem(k_vals,a,basefmt='black')
    plt.title("Коефіцієнти $a_k$ (Частотна область)")
    plt.xlabel("k")
    plt.ylabel("$a_k$")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.stem(k_vals,b,basefmt='black')
    plt.title("Коефіцієнти $b_k$ (Частотна область)")
    plt.xlabel("k")
    plt.ylabel("$b_k$")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def calc_relative_error(y_exact,y_approx):
    error_norm=np.linalg.norm(y_exact-y_approx)
    exact_norm=np.linalg.norm(y_exact)
    if exact_norm==0:
        return 0.0
    return (error_norm/exact_norm)*100

def save_to_file(filename,N,a,b,error):
    with open(filename,"w",encoding="utf-8") as f:
        f.write(f"Порядок наближення N: {N}\n")
        f.write("-"*30+"\n")
        f.write("Коефіцієнти тригонометричного ряду Фурє:\n")
        for k in range(N+1):
            f.write(f"k={k:2d} | a_{k} = {a[k]:.6f} | b_{k} = {b[k]:.6f}\n")
        f.write("-"*30+"\n")
        f.write(f"Відносна похибка наближення: {error:.4f}%\n")

def main():
    N = 10
    
    print(f"Обчислення коефіцієнтів для N={N}...")
    a_coeffs = [calc_coefficient(k) for k in range(N + 1)]
    b_coeffs = [calc_coefficient_bk(k) for k in range(N + 1)]
    
    x_vals = np.linspace(0, interval, 1000)
    y_exact = exact_f(x_vals)
    y_approx = np.array([fourier_approx(x, a_coeffs, b_coeffs, N) for x in x_vals])
    
    error = calc_relative_error(y_exact, y_approx)
    save_to_file('fourier_results.txt', N, a_coeffs, b_coeffs, error)
    
    print(f"Готово! Відносна похибка: {error:.2f}%")
    print("Результати збережено у файл 'fourier_results.txt'")

    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, y_exact, label="Оригінальна функція $f(x)$", color="blue", alpha=0.7)
    plt.plot(x_vals, y_approx, label=f"Наближення (N={N})", color="red", linestyle="--")
    plt.title(f"Наближення функції $f(x)=19 \sin(19\pi x)$ рядом Фур'є на $[0, \pi]$")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_garmonics(a_coeffs, b_coeffs, N)

if __name__=="__main__":
    main()
