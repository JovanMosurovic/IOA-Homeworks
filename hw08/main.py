import numpy as np
import matplotlib.pyplot as plt
import sys
from typing import Tuple, List
import time


class ProjectInfo:
    COURSE = "INZENJERSKI OPTIMIZACIONI ALGORITMI"
    COURSE_EN = "ENGINEERING OPTIMIZATION ALGORITHMS"
    ASSIGNMENT = "Homework Assignment 8"
    AUTHOR = "Jovan Mosurovic"
    INDEX = "2022/0589"
    SEPARATOR = "=" * 80
    SEPARATOR_SMALL = "-" * 80
    PROBLEM = "ALLOCATION OF PROCESSES TO CPU CORES - GENETIC ALGORITHM"


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
    Optimizaciona funkcija: maksimalno vreme izvrsavanja izmedju jezgara

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


def generate_initial_population(n_processes: int, pop_size: int) -> np.ndarray:
    """
    Generisanje pocetne populacije

    Args:
        n_processes: Broj procesa
        pop_size: Velicina populacije

    Returns:
        Matrica dimenzija (pop_size, n_processes)
    """
    return np.random.randint(1, 5, size=(pop_size, n_processes))


def evaluate_population(population: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Evaluacija cele populacije

    Args:
        population: Matrica populacije
        t: Trajanja procesa

    Returns:
        Vektor fitness vrednosti
    """
    fitness = np.zeros(len(population))
    for i in range(len(population)):
        fitness[i] = objective_function(population[i], t)
    return fitness


def tournament_selection(population: np.ndarray, fitness: np.ndarray,
                          tournament_size: int = 3) -> np.ndarray:
    """
    Turnirska selekcija

    Args:
        population: Matrica populacije
        fitness: Vektor fitness vrednosti
        tournament_size: Velicina turnira

    Returns:
        Selektovana individua
    """
    indices = np.random.randint(0, len(population), size=tournament_size)
    tournament_fitness = fitness[indices]
    winner_idx = indices[np.argmin(tournament_fitness)]
    return population[winner_idx].copy()


def crossover_k_genes(parent1: np.ndarray, parent2: np.ndarray,
                      k: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ukrstanje sa k gena (prilagodjeno za TSP-like probleme)

    Args:
        parent1: Prvi roditelj
        parent2: Drugi roditelj
        k: Broj gena za ukrstanje (None = slucajan izbor)

    Returns:
        Dva potomka
    """
    n = len(parent1)
    if k is None:
        k = np.random.randint(1, max(2, n // 3))

    child1 = parent2.copy()
    child2 = parent1.copy()

    # Biramo k slucajnih pozicija
    positions = np.random.choice(n, size=k, replace=False)

    # Menjamo gene na tim pozicijama
    for pos in positions:
        child1[pos] = parent1[pos]
        child2[pos] = parent2[pos]

    return child1, child2


def crossover_one_point(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ukrstanje sa jednom tackom preseka

    Args:
        parent1: Prvi roditelj
        parent2: Drugi roditelj

    Returns:
        Dva potomka
    """
    n = len(parent1)
    point = np.random.randint(1, n)

    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])

    return child1, child2


def mutate(individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """
    Mutacija individue

    Args:
        individual: Individua za mutaciju
        mutation_rate: Verovatnoca mutacije po genu

    Returns:
        Mutirana individua
    """
    mutated = individual.copy()
    n = len(individual)

    for i in range(n):
        if np.random.random() < mutation_rate:
            mutated[i] = np.random.randint(1, 5)

    return mutated


def genetic_algorithm(
        t: np.ndarray,
        pop_size: int = 2000,
        n_generations: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism: int = 10,
        show_progress: bool = False
) -> Tuple[np.ndarray, List[float]]:
    """
    Implementacija genetickog algoritma

    Args:
        t: Vremena trajanja procesa
        pop_size: Velicina populacije
        n_generations: Broj generacija
        crossover_rate: Verovatnoca ukrstanja
        mutation_rate: Verovatnoca mutacije
        elitism: Broj elitnih jedinki
        show_progress: Da li prikazati napredak

    Returns:
        Najbolje resenje i istorija optimizacione funkcije
    """
    n_processes = len(t)

    # Pocetna populacija
    population = generate_initial_population(n_processes, pop_size)
    fitness = evaluate_population(population, t)

    # Najbolje resenje
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    # Istorija
    history = [best_fitness]

    for generation in range(n_generations):
        # Nova populacija
        new_population = []

        # Elitizam - zadrzavamo najbolje jedinke
        elite_indices = np.argsort(fitness)[:elitism]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Generisanje ostatka populacije
        while len(new_population) < pop_size:
            # Selekcija roditelja
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)

            # Ukrstanje
            if np.random.random() < crossover_rate:
                # child1, child2 = crossover_one_point(parent1, parent2)
                child1, child2 = crossover_k_genes(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutacija
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        # Azuriranje populacije
        population = np.array(new_population)
        fitness = evaluate_population(population, t)

        # Azuriranje najboljeg resenja
        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]

        if current_best_fitness < best_fitness:
            best_solution = population[current_best_idx].copy()
            best_fitness = current_best_fitness

        history.append(best_fitness)

        if show_progress and generation % 10 == 0:
            print(f"Generacija {generation:3d}: Best = {best_fitness:.0f} µs")

    return best_solution, history


def multiple_runs(
        t: np.ndarray,
        n_runs: int = 20,
        pop_size: int = 2000,
        n_generations: int = 50
) -> Tuple[List[List[float]], np.ndarray, float, List[float]]:
    """
    Visestruka nezavisna pokretanja optimizacije

    Args:
        t: Vremena trajanja procesa
        n_runs: Broj nezavisnih pokretanja
        pop_size: Velicina populacije
        n_generations: Broj generacija

    Returns:
        Lista istorija za svako pokretanje, najbolje resenje, najbolja vrednost, sve najbolje vrednosti
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
        x_best, history = genetic_algorithm(
            t, pop_size, n_generations,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elitism=10,
            show_progress=False
        )
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
        iterations = np.arange(len(cum_min)) * 2000  # pop_size = 2000
        plt.semilogx(iterations, cum_min, alpha=0.6, linewidth=1.5,
                     label=f'Pokretanje {i + 1}')

    plt.xlabel('Broj iteracija (evaluacija funkcije)', fontsize=13)
    plt.ylabel('Kumulativni minimum (µs)', fontsize=13)
    plt.title('Kumulativni minimumi za sva pokretanja - Geneticki algoritam',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
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
            padded = np.pad(cum_min, (0, max_len - len(cum_min)), mode='edge')
        else:
            padded = cum_min
        padded_histories.append(padded)

    # Racunanje srednje vrednosti
    avg_best = np.mean(padded_histories, axis=0)
    iterations = np.arange(len(avg_best)) * 2000  # pop_size = 2000

    plt.figure(figsize=(14, 8))
    plt.semilogx(iterations, avg_best, linewidth=2.5, color='darkblue',
                 label='Srednje najbolje resenje')

    plt.xlabel('Broj iteracija (evaluacija funkcije)', fontsize=13)
    plt.ylabel('Srednje najbolje resenje (µs)', fontsize=13)
    plt.title('Srednje najbolje pronadjeno resenje - Geneticki algoritam',
              fontsize=14, fontweight='bold')
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

        f.write("NAJBOLJE PRONADJENO RESENJE - GENETICKI ALGORITAM\n")
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

    print(f"\n[2] PARAMETRI ALGORITMA")
    print(ProjectInfo.SEPARATOR_SMALL)
    pop_size = 2000
    n_generations = 50
    print(f"Velicina populacije: {pop_size}")
    print(f"Broj generacija: {n_generations}")
    print(f"Verovatnoca ukrstanja: 0.8")
    print(f"Verovatnoca mutacije: 0.1")
    print(f"Elitizam: 10 jedinki")
    print(f"Strategija selekcije: Turnirska (velicina turnira = 3)")
    print(f"Tip ukrstanja: k-gena")
    print(f"Broj nezavisnih pokretanja: 20")
    print(f"Ukupan broj evaluacija po pokretanju: {pop_size * (n_generations + 1):,}")
    print(f"Ukupan broj evaluacija: {pop_size * (n_generations + 1) * 20:,}")

    # Visestruka pokretanja
    all_histories, best_solution, best_value, all_best_values = multiple_runs(
        t, n_runs=20, pop_size=pop_size, n_generations=n_generations
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