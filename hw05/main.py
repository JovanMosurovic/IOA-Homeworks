import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys


class ProjectInfo:
    COURSE = "INZENJERSKI OPTIMIZACIONI ALGORITMI"
    COURSE_EN = "ENGINEERING OPTIMIZATION ALGORITHMS"
    ASSIGNMENT = "Homework Assignment 5"
    AUTHOR = "Jovan Mosurovic"
    INDEX = "2022/0589"
    SEPARATOR = "=" * 70
    SEPARATOR_SMALL = "-" * 70
    PROBLEM = "NEURAL NETWORK FUNCTION APPROXIMATION"


def print_header():
    print(ProjectInfo.SEPARATOR)
    print(ProjectInfo.COURSE)
    print(f"({ProjectInfo.COURSE_EN})")
    print(ProjectInfo.ASSIGNMENT)
    print(ProjectInfo.SEPARATOR_SMALL)
    print(f"Author: {ProjectInfo.AUTHOR} ({ProjectInfo.INDEX})")
    print(ProjectInfo.SEPARATOR)
    print()
    print(ProjectInfo.PROBLEM)
    print(ProjectInfo.SEPARATOR)


def load_data(filename='data.txt'):
    print("\n[1] Ucitavanje trening podataka...")
    print(ProjectInfo.SEPARATOR_SMALL)

    data = np.loadtxt(filename)
    x_train = data[:, 0]
    y_train = data[:, 1]

    print(f"Fajl: {filename}")
    print(f"Ucitano {len(x_train)} odbiraka")
    print(f"  Opseg x: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"  Opseg y: [{y_train.min():.3f}, {y_train.max():.3f}]")

    return x_train, y_train


def neural_network(x_in, w):
    hidden = np.tanh(w[0:5] * x_in + w[5:10])
    y_out = np.dot(w[10:15], hidden) + w[15]
    return y_out


def mean_square_error(w, x_train, y_train):
    y_pred = np.array([neural_network(x, w) for x in x_train])
    return np.mean((y_train - y_pred) ** 2)


def optimize_network(x_train, y_train, target_mse=5e-4):
    print("\n[2] Optimizacija neuronske mreze...")
    print(ProjectInfo.SEPARATOR_SMALL)
    print("Metoda: Nelder-Mead simplex")
    print("Ogranicenja: -10 <= w <= 10")
    print(f"Cilj: MSE < {target_mse:.1e}")
    print()

    # Pocetne vrednosti
    np.random.seed(42)
    w_initial = np.random.uniform(-2, 2, 16)

    initial_mse = mean_square_error(w_initial, x_train, y_train)
    print(f"Pocetna MSE: {initial_mse:.6e}")
    print()

    # Progress
    progress = {'iter': 0, 'best_mse': initial_mse, 'last_update': 0}

    def callback(xk):
        progress['iter'] += 1
        current_mse = mean_square_error(xk, x_train, y_train)

        if current_mse < progress['best_mse']:
            progress['best_mse'] = current_mse

        if progress['iter'] % 200 == 0 or progress['iter'] == 1:
            max_iter = 50000
            percentage = min(100, (progress['iter'] / max_iter) * 100)
            bar_length = 40
            filled = int(bar_length * percentage / 100)
            bar = '█' * filled + '░' * (bar_length - filled)

            # Ispis progress bar-a
            sys.stdout.write(
                f"\r[{bar}] {percentage:5.1f}% | Iter: {progress['iter']:5d} | Best MSE: {progress['best_mse']:.8e}")
            sys.stdout.flush()
            progress['last_update'] = progress['iter']

    print("Napredak optimizacije:")
    print(ProjectInfo.SEPARATOR_SMALL)

    # Nelder-Mead optimizacija
    result = minimize(
        mean_square_error,
        w_initial,
        args=(x_train, y_train),
        method='Nelder-Mead',
        bounds=[(-10, 10)] * 16,
        callback=callback,
        options={
            'maxiter': 50000,
            'xatol': 1e-8,
            'fatol': 1e-8,
            'adaptive': True,
            'disp': False
        }
    )

    bar = '█' * 40
    sys.stdout.write(f"\r[{bar}] 100.0% | Iter: {progress['iter']:5d} | Best MSE: {progress['best_mse']:.8e}")
    sys.stdout.flush()
    print("\n")

    return result.x, result.fun, result.nit, result.nfev


def print_results(w_optimal, mse_final, n_iter, n_eval, target_mse=5e-4):
    print("[3] Rezultati optimizacije")
    print(ProjectInfo.SEPARATOR)
    print(f"Finalna MSE:     {mse_final:.15e}")
    print(f"Ciljna MSE:      < {target_mse:.1e}")
    print(f"Broj iteracija:  {n_iter}")
    print(f"Broj evaluacija: {n_eval}")

    print("\nOptimalni tezinski koeficijenti (w1..w16):")
    print(ProjectInfo.SEPARATOR_SMALL)
    for i in range(16):
        print(f"w{i + 1:2d} = {w_optimal[i]:20.15f}")

def save_results(w_optimal, mse_final, target_mse=5e-4):
    print("\n[4] Cuvanje rezultata...")
    print(ProjectInfo.SEPARATOR_SMALL)

    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(ProjectInfo.SEPARATOR + "\n")
        f.write(ProjectInfo.COURSE + "\n")
        f.write(f"({ProjectInfo.COURSE_EN})\n")
        f.write(ProjectInfo.ASSIGNMENT + "\n")
        f.write(f"Author: {ProjectInfo.AUTHOR} ({ProjectInfo.INDEX})\n")
        f.write(ProjectInfo.SEPARATOR + "\n\n")

        f.write("NEURAL NETWORK OPTIMIZATION RESULTS\n")
        f.write(ProjectInfo.SEPARATOR + "\n\n")

        f.write("Method: Nelder-Mead Simplex Algorithm\n\n")

        f.write("Optimalni tezinski koeficijenti:\n")
        f.write("[")
        for i in range(16):
            f.write(f"{w_optimal[i]:.15f}")
            if i < 15:
                f.write(",")
        f.write("]\n\n")

        f.write(f"Minimalna pronadjena MSE: {mse_final:.15e}\n")
        f.write(f"Ciljna MSE: < {target_mse:.1e}\n")
        f.write(f"Status: {'USPEH' if mse_final < target_mse else 'NEUSPEH'}\n")
        f.write(ProjectInfo.SEPARATOR + "\n")

    print("Rezultati sacuvani u 'result.txt'")

def plot_results(x_train, y_train, w_optimal, mse_final):
    print("\n[5] Generisanje grafika...")
    print(ProjectInfo.SEPARATOR_SMALL)

    # Izlaz mreze
    y_output = np.array([neural_network(x, w_optimal) for x in x_train])

    # Statistika greske
    errors = np.abs(y_train - y_output)
    print(f"Maksimalna greska: {np.max(errors):.6e}")
    print(f"Prosecna greska:   {np.mean(errors):.6e}")
    print(f"RMS greska:        {np.sqrt(np.mean(errors ** 2)):.6e}")

    # Crtanje grafika
    plt.figure(figsize=(12, 7))
    plt.plot(x_train, y_train, 'bo-', linewidth=2, markersize=4,
             label='y_training(x)', alpha=0.7)
    plt.plot(x_train, y_output, 'r--', linewidth=2.5,
             label='y_out(x)', alpha=0.9)

    plt.xlabel('x', fontsize=13)
    plt.ylabel('y', fontsize=13)
    plt.title(f'Aproksimacija neuronskom mrezom | MSE = {mse_final:.6e}',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    plt.tight_layout()

    plt.savefig('graph.png', dpi=300, bbox_inches='tight')
    print("Grafik sacuvan kao 'graph.png'")

    plt.show()


def main():
    print_header()

    # Ucitavanje podataka
    x_train, y_train = load_data('data.txt')

    # Optimizacija
    w_optimal, mse_final, n_iter, n_eval = optimize_network(x_train, y_train)

    # Ispis rezultata
    print()
    print_results(w_optimal, mse_final, n_iter, n_eval)

    # Cuvanje rezultata
    save_results(w_optimal, mse_final)

    # Graficki prikaz
    plot_results(x_train, y_train, w_optimal, mse_final)

    print()
    print(ProjectInfo.SEPARATOR)
    print("OPTIMIZACIJA ZAVRSENA")
    print(ProjectInfo.SEPARATOR)
    print("\nGenerisani fajlovi:")
    print("-> result.txt    - Fajl sa rezultatima")
    print("-> graph.png     - Grafik aproksimacije")
    print(ProjectInfo.SEPARATOR)


if __name__ == "__main__":
    main()