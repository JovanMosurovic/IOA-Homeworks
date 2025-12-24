import numpy as np
import matplotlib.pyplot as plt
import sys
from typing import Tuple, List
import time


class ProjectInfo:
    COURSE = "INZENJERSKI OPTIMIZACIONI ALGORITMI"
    COURSE_EN = "ENGINEERING OPTIMIZATION ALGORITHMS"
    ASSIGNMENT = "Homework Assignment 7"
    AUTHOR = "Jovan Mosurovic"
    INDEX = "2022/0589"
    SEPARATOR = "=" * 80
    SEPARATOR_SMALL = "-" * 80
    PROBLEM = "ALLOCATION OF PROCESSES TO CPU CORES"


def print_header():
    print(ProjectInfo.SEPARATOR)
    print(ProjectInfo.COURSE)
    print(f"({ProjectInfo.COURSE_EN})")
    print(ProjectInfo.ASSIGNMENT)
    print(ProjectInfo.SEPARATOR_SMALL)
    print(f"Autor: {ProjectInfo.AUTHOR} ({ProjectInfo.INDEX})")
    print(ProjectInfo.SEPARATOR)
    print()
    print(ProjectInfo.PROBLEM)
    print(ProjectInfo.SEPARATOR)


def load_data() -> np.ndarray:
    print("\n[1] UCITAVANJE PODATAKA")
    print(ProjectInfo.SEPARATOR_SMALL)

    t = np.array([227, 316, 797, 676, 391, 332, 598, 186, 672, 941, 248,
                  948, 667, 95, 441, 886, 697, 326, 733, 220, 81, 159, 340,
                  465, 266, 815, 193, 129, 91, 598, 854, 601, 931, 724,
                  860, 929, 546, 937, 494, 273, 451, 665, 330, 903, 257,
                  339, 258, 355, 5, 628, 282, 68, 616, 176, 304, 440, 150,
                  217, 474, 476, 255, 297, 279, 260, 482, 211, 495, 246,
                  838, 180, 862, 178, 750, 611, 209, 759, 249, 85, 618,
                  536, 634, 174, 248, 684, 80, 875, 428, 618, 313, 178, 9,
                  210, 870, 972, 441, 378, 275, 966, 58, 408])

    print(f"Broj procesa: {len(t)}")
    print(f"Broj jezgara: 4")
    print(f"Ukupno vreme svih procesa: {np.sum(t)} µs")
    print(f"Prosecno vreme procesa: {np.mean(t):.2f} µs")
    print(f"Optimalna donja granica: {np.sum(t) / 4:.2f} µs")
    print(f"Min/Max vreme procesa: {np.min(t)} / {np.max(t)} µs")

    return t


def objective_function(x: np.ndarray, t: np.ndarray) -> float:
    """
    Optimizaciona funkcija: maksimalno vreme izvrsavanja medu jezgrima

    Args:
        x: Vektor dodeljivanja procesa jezgrima (vrednosti 1-4)
        t: Vremena trajanja procesa

    Returns:
        Maksimalno vreme izvrsavanja
    """
    core_times = np.zeros(4)
    for i in range(len(x)):
        core_times[int(x[i]) - 1] += t[i]
    return np.max(core_times)


def generate_initial_solution(n_processes: int) -> np.ndarray:
    return np.random.randint(1, 5, n_processes)


def get_next(x: np.ndarray, max_changes: int) -> np.ndarray:
    """
    Generisanje susednog resenja promenom rasporeda procesa

    Args:
        x: Trenutno resenje
        max_changes: Maksimalan broj promena

    Returns:
        Novo susedno resenje
    """
    x_new = x.copy()
    n_changes = np.random.randint(1, max_changes + 1)

    for _ in range(n_changes):
        idx = np.random.randint(0, len(x))
        new_core = np.random.randint(1, 5)
        x_new[idx] = new_core

    return x_new


def simulated_annealing(
        t: np.ndarray,
        T0: float = 50000,
        alpha: float = 0.95,
        max_iter: int = 20000,
        initial_solution: np.ndarray = None,
        show_progress: bool = False
) -> Tuple[np.ndarray, List[float]]:
    """
    Implementacija algoritma simuliranog kaljenja

    Args:
        t: Vremena trajanja procesa
        T0: Pocetna temperatura
        alpha: Koeficijent hladenja
        max_iter: Maksimalan broj iteracija
        initial_solution: Pocetno resenje (ako postoji)
        show_progress: Da li prikazati napredak

    Returns:
        Najbolje resenje i istorija optimizacione funkcije
    """
    n_processes = len(t)

    # Pocetno resenje
    if initial_solution is None:
        x_current = generate_initial_solution(n_processes)
    else:
        x_current = initial_solution.copy()

    f_current = objective_function(x_current, t)

    # Najbolje resenje
    x_best = x_current.copy()
    f_best = f_current

    # Istorija
    history = [f_current]

    # Temperatura
    T = T0

    for i in range(max_iter):
        # Smanjivanje broja promena sa temperaturom
        max_changes = max(1, int(10 * T / T0))

        # Generisanje suseda
        x_new = get_next(x_current, max_changes)
        f_new = objective_function(x_new, t)

        # Promena energije
        delta_E = f_new - f_current

        # Odluka o prihvatanju
        if delta_E < 0:
            # Bolje resenje - prihvati
            x_current = x_new
            f_current = f_new

            if f_new < f_best:
                x_best = x_new.copy()
                f_best = f_new
        else:
            # Gore resenje - prihvati sa verovatnocom
            p = np.exp(-delta_E / T)
            if np.random.random() < p:
                x_current = x_new
                f_current = f_new

        history.append(f_best)

        # Hladjenje
        T = alpha * T

    return x_best, history


def repeated_annealing(
        t: np.ndarray,
        n_repeats: int = 10,
        T0: float = 50000,
        max_iter: int = 10000
) -> Tuple[np.ndarray, List[float]]:
    """
    Ponovljeno kaljenje (reannealing)

    Args:
        t: Vremena trajanja procesa
        n_repeats: Broj ponavljanja
        T0: Pocetna temperatura
        max_iter: Maksimalan broj iteracija po ponavljanju

    Returns:
        Najbolje resenje i kombinovana istorija
    """
    x_best = None
    f_best = float('inf')
    combined_history = []

    for repeat in range(n_repeats):
        x_current, history = simulated_annealing(
            t, T0, 0.95, max_iter, x_best, show_progress=False
        )

        combined_history.extend(history)

        f_current = objective_function(x_current, t)
        if f_current < f_best:
            x_best = x_current.copy()
            f_best = f_current

    return x_best, combined_history


def multiple_runs(
        t: np.ndarray,
        n_runs: int = 20,
        n_repeats: int = 10,
        T0: float = 50000,
        max_iter: int = 10000
) -> Tuple[List[List[float]], np.ndarray, float, List[float]]:
    """
    Visestruka nezavisna pokretanja optimizacije

    Args:
        t: Vremena trajanja procesa
        n_runs: Broj nezavisnih pokretanja
        n_repeats: Broj ponavljanja po pokretanju
        T0: Pocetna temperatura
        max_iter: Maksimalan broj iteracija

    Returns:
        Lista istorija za svako pokretanje
    """
    print(f"\n[3] VISESTRUKA POKRETANJA ({n_runs} pokretanja)")
    print(ProjectInfo.SEPARATOR_SMALL)

    all_histories = []
    all_best_solutions = []
    all_best_values = []

    start_time = time.time()

    print("\nNapredak:")
    print(ProjectInfo.SEPARATOR_SMALL)

    for run in range(n_runs):
        x_best, history = repeated_annealing(t, n_repeats, T0, max_iter)
        f_best = objective_function(x_best, t)

        all_histories.append(history)
        all_best_solutions.append(x_best)
        all_best_values.append(f_best)

        # Progress bar
        percentage = (run + 1) / n_runs * 100
        bar_length = 40
        filled = int(bar_length * percentage / 100)
        bar = '█' * filled + '░' * (bar_length - filled)

        sys.stdout.write(
            f"\r[{bar}] {percentage:5.1f}% | Pokretanje: {run + 1:2d}/{n_runs} | "
            f"Best: {f_best:5.0f} µs | Global best: {np.min(all_best_values):5.0f} µs"
        )
        sys.stdout.flush()

    print()

    elapsed_time = time.time() - start_time

    # Statistika
    print(f"\n{ProjectInfo.SEPARATOR_SMALL}")
    print("STATISTIKA SVIH POKRETANJA:")
    print(ProjectInfo.SEPARATOR_SMALL)
    print(f"Ukupno vreme izvrsavanja: {elapsed_time:.2f} s")
    print(f"Najbolje resenje: {np.min(all_best_values):.0f} µs")
    print(f"Najgore resenje: {np.max(all_best_values):.0f} µs")
    print(f"Prosecno resenje: {np.mean(all_best_values):.2f} µs")
    print(f"Standardna devijacija: {np.std(all_best_values):.2f} µs")

    # Najbolje ukupno resenje
    best_idx = np.argmin(all_best_values)
    global_best_solution = all_best_solutions[best_idx]
    global_best_value = all_best_values[best_idx]

    return all_histories, global_best_solution, global_best_value, all_best_values


def cumulative_minimum(history: List[float]) -> np.ndarray:
    cum_min = np.zeros(len(history))
    cum_min[0] = history[0]

    for i in range(1, len(history)):
        cum_min[i] = min(cum_min[i - 1], history[i])

    return cum_min


def plot_cumulative_minimums(all_histories: List[List[float]]):
    print("\n[4] GENERISANJE GRAFIKA")
    print(ProjectInfo.SEPARATOR_SMALL)

    plt.figure(figsize=(14, 8))

    for i, history in enumerate(all_histories):
        cum_min = cumulative_minimum(history)
        plt.plot(cum_min, alpha=0.6, linewidth=1.5, label=f'Pokretanje {i + 1}')

    plt.xlabel('Broj iteracija', fontsize=13)
    plt.ylabel('Kumulativni minimum (µs)', fontsize=13)
    plt.title('Kumulativni minimumi za sva pokretanja', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()

    plt.savefig('cumulative_minimums_graph.png', dpi=300, bbox_inches='tight')
    print("Grafik kumulativnih minimuma sacuvan: cumulative_minimums_graph.png")
    plt.close()


def plot_average_best(all_histories: List[List[float]]):
    max_len = max(len(h) for h in all_histories)

    padded_histories = []
    for history in all_histories:
        cum_min = cumulative_minimum(history)
        if len(cum_min) < max_len:
            padded = np.pad(cum_min, (0, max_len - len(cum_min)),
                            mode='edge')
        else:
            padded = cum_min
        padded_histories.append(padded)

    # Racunanje srednje vrednosti
    avg_best = np.mean(padded_histories, axis=0)

    plt.figure(figsize=(14, 8))
    plt.semilogx(avg_best, linewidth=2.5, color='darkblue', label='Srednje najbolje resenje')

    plt.xlabel('Broj iteracija', fontsize=13)
    plt.ylabel('Srednje najbolje resenje (µs)', fontsize=13)
    plt.title('Srednje najbolje pronadjeno resenje', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig('average_best_graph.png', dpi=300, bbox_inches='tight')
    print("Grafik srednje najbolje vrednosti sacuvan: average_best_graph.png")
    plt.close()


def analyze_solution(x: np.ndarray, t: np.ndarray):
    print(f"\n[5] ANALIZA NAJBOLJEG RESENJA")
    print(ProjectInfo.SEPARATOR_SMALL)

    core_times = np.zeros(4)
    core_processes = [[] for _ in range(4)]

    for i in range(len(x)):
        core_idx = int(x[i]) - 1
        core_times[core_idx] += t[i]
        core_processes[core_idx].append(i)

    print("\nRaspodela po jezgrima:")
    for i in range(4):
        print(f"  Jezgro {i + 1}: {len(core_processes[i]):2d} procesa, "
              f"vreme = {core_times[i]:7.0f} µs")

    print(f"\nMaksimalno vreme (optimizaciona funkcija): {np.max(core_times):.0f} µs")
    print(f"Minimalno vreme: {np.min(core_times):.0f} µs")
    print(f"Prosecno vreme: {np.mean(core_times):.2f} µs")
    print(f"Standardna devijacija: {np.std(core_times):.2f} µs")
    print(f"Balansiranost: {(1 - np.std(core_times) / np.mean(core_times)) * 100:.2f}%")


def save_solution(x: np.ndarray, f_best: float, all_best_values: List[float]):
    print(f"\n[6] CUVANJE REZULTATA")
    print(ProjectInfo.SEPARATOR_SMALL)

    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(ProjectInfo.SEPARATOR + "\n")
        f.write(ProjectInfo.COURSE + "\n")
        f.write(f"({ProjectInfo.COURSE_EN})\n")
        f.write(ProjectInfo.ASSIGNMENT + "\n")
        f.write(f"Autor: {ProjectInfo.AUTHOR} ({ProjectInfo.INDEX})\n")
        f.write(ProjectInfo.SEPARATOR + "\n\n")

        f.write("NAJBOLJE PRONADJENO RESENJE\n")
        f.write(ProjectInfo.SEPARATOR + "\n\n")

        f.write("x=(")
        for i in range(len(x)):
            f.write(f"{int(x[i])}")
            if i < len(x) - 1:
                f.write(", ")
        f.write(")\n\n")

        f.write(f"Minimalna vrednost optimizacione funkcije: {f_best:.0f} µs\n\n")

        f.write("\n" + ProjectInfo.SEPARATOR + "\n")

    print("Rezultati sacuvani u: result.txt")


def main():
    print_header()

    # Ucitavanje podataka
    t = load_data()

    # Parametri algoritma
    print(f"\n[2] PARAMETRI ALGORITMA")
    print(ProjectInfo.SEPARATOR_SMALL)
    T0 = 100 * 500
    print(f"Pocetna temperatura T0: {T0}")
    print(f"Koeficijent hladenja α: 0.95")
    print(f"Maksimalan broj iteracija po pokretanju: 10,000")
    print(f"Broj ponavljanja (reannealing): 10")
    print(f"Broj nezavisnih pokretanja: 20")
    print(f"Ukupan broj evaluacija: {10 * 10000 * 20:,}")

    # Visestruka pokretanja
    all_histories, best_solution, best_value, all_best_values = multiple_runs(
        t, n_runs=20, n_repeats=10, T0=T0, max_iter=10000
    )

    # Analiza resenja
    analyze_solution(best_solution, t)

    # Crtanje grafika
    plot_cumulative_minimums(all_histories)
    plot_average_best(all_histories)

    # Cuvanje rezultata
    save_solution(best_solution, best_value, all_best_values)

    print()
    print(ProjectInfo.SEPARATOR)
    print("OPTIMIZACIJA ZAVRSENA")
    print(ProjectInfo.SEPARATOR)
    print("\nGenerisani fajlovi:")
    print("  -> result.txt                          - Rezultati optimizacije")
    print("  -> cumulative_minimums_graph.png       - Kumulativni minimumi")
    print("  -> average_best_graph.png              - Srednje najbolje resenje")
    print(ProjectInfo.SEPARATOR)


if __name__ == "__main__":
    main()