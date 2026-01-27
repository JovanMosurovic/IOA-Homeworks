import numpy as np
import random
import sys
import time


class ProjectInfo:
    COURSE = "INZENJERSKI OPTIMIZACIONI ALGORITMI"
    COURSE_EN = "ENGINEERING OPTIMIZATION ALGORITHMS"
    ASSIGNMENT = "Homework Assignment 9"
    AUTHOR = "Jovan Mosurovic"
    INDEX = "2022/0589"
    SEPARATOR = "=" * 80
    SEPARATOR_SMALL = "-" * 80
    PROBLEM = "SIGNAL SOURCE LOCALIZATION - DIFFERENTIAL EVOLUTION"


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


S_measured = np.array([
    -0.09940112332480822, -0.09570265063923192, -0.07782620994584906,
    -0.044595775065571636, -0.008470411838648773, -0.0013292572938769093,
    -0.01402876134848341, 0.0011785680597112547, -0.0016096599564817682,
    -0.03141072397571561, -0.05773121434057853, -0.07098734083487862,
    -0.07421256224434619, -0.09674779542915338, -0.13216942328836218,
    -0.18406033359301877, -0.24214426775005213, -0.25978279767024376,
    -0.2186443973931424, -0.3289283483195699, -0.4205252223787085,
    -0.32130499477499636, -0.205134242990832, -0.13760381018149595
])

measurement_points = []
for i in range(7):
    measurement_points.append((-15 + i * 5, -15))
for i in range(6):
    measurement_points.append((15, -10 + i * 5))
for i in range(6):
    measurement_points.append((10 - i * 5, 15))
for i in range(5):
    measurement_points.append((-15, 10 - i * 5))
measurement_points = np.array(measurement_points)


def load_data():
    print("\n[1] UCITAVANJE PODATAKA")
    print(ProjectInfo.SEPARATOR_SMALL)
    print(f"Broj mernih tacaka: {len(S_measured)}")
    print(f"Broj nepoznatih izvora: 2")
    print(f"Raspon kvadrata: [-15, 15] x [-15, 15] m")
    print(f"Razmak izmedju mernih tacaka: 5 m")
    print(f"Opseg izmerenog signala: [{np.min(S_measured):.6f}, {np.max(S_measured):.6f}]")
    print(f"Prosecna vrednost signala: {np.mean(S_measured):.6f}")


def objective_function(x):
    x_P1, y_P1, x_P2, y_P2, A1, A2 = x

    error_sum = 0.0
    for i in range(len(S_measured)):
        x_i, y_i = measurement_points[i]

        r1 = np.sqrt((x_i - x_P1) ** 2 + (y_i - y_P1) ** 2)
        r2 = np.sqrt((x_i - x_P2) ** 2 + (y_i - y_P2) ** 2)

        S_calc = (A1 / r1 if r1 > 1e-10 else 0) + (A2 / r2 if r2 > 1e-10 else 0)
        error_sum += (S_calc - S_measured[i]) ** 2

    return error_sum


class DifferentialEvolution:

    def __init__(self, func, bounds, pop_size=100, F=0.8, CR=0.9,
                 max_iter=5000, tol=1e-14, seed=None):
        self.func = func
        self.bounds = np.array(bounds)
        self.D = len(bounds)
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.max_iter = max_iter
        self.tol = tol

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def initialize(self):
        self.population = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.pop_size, self.D)
        )
        self.fitness = np.array([self.func(ind) for ind in self.population])

        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

    def mutate_and_crossover(self, idx):
        candidates = list(range(self.pop_size))
        candidates.remove(idx)
        a, b, c = random.sample(candidates, 3)

        z = self.population[a] + self.F * (self.population[b] - self.population[c])

        R = random.randint(0, self.D - 1)
        y = np.zeros(self.D)
        for i in range(self.D):
            if random.random() < self.CR or i == R:
                y[i] = z[i]
            else:
                y[i] = self.population[idx][i]

        y = np.clip(y, self.bounds[:, 0], self.bounds[:, 1])
        return y

    def optimize(self):
        self.initialize()

        for iteration in range(1, self.max_iter + 1):
            new_population = []
            new_fitness = []

            for k in range(self.pop_size):
                y = self.mutate_and_crossover(k)
                f_y = self.func(y)

                if f_y < self.fitness[k]:
                    new_population.append(y)
                    new_fitness.append(f_y)
                else:
                    new_population.append(self.population[k])
                    new_fitness.append(self.fitness[k])

            self.population = np.array(new_population)
            self.fitness = np.array(new_fitness)

            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_solution = self.population[current_best_idx].copy()
                self.best_fitness = self.fitness[current_best_idx]

            if self.best_fitness <= self.tol:
                return self.best_solution, self.best_fitness, iteration

        return self.best_solution, self.best_fitness, self.max_iter


def multiple_runs(bounds, n_runs=5, pop_size=100, max_iter=5000):
    print(f"\n[3] VISESTRUKA POKRETANJA ({n_runs} pokretanja)")
    print(ProjectInfo.SEPARATOR_SMALL)

    seeds = [17, 89, 234, 567, 891, 1024, 2048, 3141, 5926, 8192][:n_runs]
    all_results = []
    all_fitness = []

    start_time = time.time()

    print("\nNapredak:")
    print(ProjectInfo.SEPARATOR_SMALL)

    for run, seed in enumerate(seeds):
        de = DifferentialEvolution(
            func=objective_function,
            bounds=bounds,
            pop_size=pop_size,
            F=0.8,
            CR=0.9,
            max_iter=max_iter,
            tol=1e-14,
            seed=seed
        )

        solution, fitness, iterations = de.optimize()
        all_results.append((solution, fitness, iterations))
        all_fitness.append(fitness)

        percentage = (run + 1) / n_runs * 100
        bar_length = 40
        filled = int(bar_length * percentage / 100)
        bar = '█' * filled + '░' * (bar_length - filled)

        sys.stdout.write(
            f"\r[{bar}] {percentage:5.1f}% | Pokretanje: {run + 1:2d}/{n_runs} | "
            f"Best: {fitness:5.2e} | Global best: {np.min(all_fitness):5.2e}"
        )
        sys.stdout.flush()

    print()

    elapsed_time = time.time() - start_time

    print(f"\n{ProjectInfo.SEPARATOR_SMALL}")
    print("STATISTIKA SVIH POKRETANJA:")
    print(ProjectInfo.SEPARATOR_SMALL)
    print(f"Ukupno vreme izvrsavanja: {elapsed_time:.2f} s")
    print(f"Najbolje resenje: {np.min(all_fitness):.6e}")
    print(f"Najgore resenje: {np.max(all_fitness):.6e}")
    print(f"Prosecno resenje: {np.mean(all_fitness):.6e}")
    print(f"Standardna devijacija: {np.std(all_fitness):.6e}")

    best_idx = np.argmin(all_fitness)
    return all_results[best_idx], all_fitness


def analyze_solution(solution, fitness, iterations):
    print(f"\n[4] ANALIZA NAJBOLJEG RESENJA")
    print(ProjectInfo.SEPARATOR_SMALL)

    x_P1, y_P1, x_P2, y_P2, A1, A2 = solution

    print("\nPronadjeni izvori signala:")
    print(f"    Izvor 1: x1 = {x_P1:.15f} ≈ {x_P1:.1f}, y1 = {y_P1:.15f} ≈ {y_P1:.1f}, A1 = {A1:.15f} ≈ {A1:.1f}")
    print(f"    Izvor 2: x2 = {x_P2:.15f} ≈ {x_P2:.1f}, y2 = {y_P2:.15f} ≈ {y_P2:.1f}, A2 = {A2:.15f} ≈ {A2:.1f}")

    print("\nPronadjeno resenje:")
    print(f"    x = ({x_P1:.15f}, {y_P1:.15f}, {x_P2:.15f}, {y_P2:.15f}, {A1:.15f}, {A2:.15f})")
    print(f"    x ≈ ({x_P1:.1f}, {y_P1:.1f}, {x_P2:.1f}, {y_P2:.1f}, {A1:.1f}, {A2:.1f})")

    print(f"\nOptimizaciona funkcija: f_opt = {fitness:.15e}")
    print(f"Broj iteracija: {iterations}")

    if fitness <= 1e-14:
        print("Postignuta trazena preciznost (f_opt <= 10^-14)")

    # r1_from_center = np.sqrt(x_P1 ** 2 + y_P1 ** 2)
    # r2_from_center = np.sqrt(x_P2 ** 2 + y_P2 ** 2)
    # distance_between = np.sqrt((x_P2 - x_P1) ** 2 + (y_P2 - y_P1) ** 2)
    #
    # print(f"\nGeometrijska svojstva:")
    # print(f"  Rastojanje izvora 1 od centra: {r1_from_center:.2f} m")
    # print(f"  Rastojanje izvora 2 od centra: {r2_from_center:.2f} m")
    # print(f"  Rastojanje izmedju izvora: {distance_between:.2f} m")


def save_solution(solution, fitness, iterations, all_fitness):
    print(f"\n[5] CUVANJE REZULTATA")
    print(ProjectInfo.SEPARATOR_SMALL)

    x_P1, y_P1, x_P2, y_P2, A1, A2 = solution

    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(ProjectInfo.SEPARATOR + "\n")
        f.write(ProjectInfo.COURSE + "\n")
        f.write(f"({ProjectInfo.COURSE_EN})\n")
        f.write(ProjectInfo.ASSIGNMENT + "\n")
        f.write(f"Autor: {ProjectInfo.AUTHOR} ({ProjectInfo.INDEX})\n")
        f.write(ProjectInfo.SEPARATOR + "\n\n")

        f.write("NAJBOLJE PRONADJENO RESENJE - DIFERENCIJALNA EVOLUCIJA\n")
        f.write(ProjectInfo.SEPARATOR + "\n\n")

        f.write(f"x=({x_P1:.15f}, {y_P1:.15f}, {x_P2:.15f}, {y_P2:.15f}, {A1:.15f}, {A2:.15f})\n\n")
        f.write(f"x≈({x_P1:.1f}, {y_P1:.1f}, {x_P2:.1f}, {y_P2:.1f}, {A1:.1f}, {A2:.1f})\n\n")

        f.write(f"Minimalna vrednost optimizacione funkcije: f_opt = {fitness:.15e}\n")
        f.write(f"Broj iteracija: {iterations}\n\n")

        if fitness <= 1e-14:
            f.write("Postignuta trazena preciznost (f_opt <= 10^-14)\n\n")

        f.write(ProjectInfo.SEPARATOR + "\n")

    print("Rezultati sacuvani u: result.txt")


def main():
    print_header()

    load_data()

    print(f"\n[2] PARAMETRI ALGORITMA")
    print(ProjectInfo.SEPARATOR_SMALL)
    pop_size = 100
    max_iter = 5000
    n_runs = 5
    print(f"Velicina populacije: {pop_size}")
    print(f"Maksimalan broj generacija: {max_iter}")
    print(f"Diferencijalni tezinski faktor F: 0.8")
    print(f"Verovatnoca ukrstanja CR: 0.9")
    print(f"Tolerancija: 10^-14")
    print(f"Broj nezavisnih pokretanja: {n_runs}")
    print(f"Ukupan broj evaluacija po pokretanju: ~{pop_size * (max_iter + 1):,}")
    print(f"Ukupan broj evaluacija: ~{pop_size * (max_iter + 1) * n_runs:,}")

    bounds = [(-15, 15), (-15, 15), (-15, 15), (-15, 15), (-15, 15), (-15, 15)]

    (best_solution, best_fitness, best_iterations), all_fitness = multiple_runs(
        bounds, n_runs=n_runs, pop_size=pop_size, max_iter=max_iter
    )

    analyze_solution(best_solution, best_fitness, best_iterations)

    save_solution(best_solution, best_fitness, best_iterations, all_fitness)

    print()
    print(ProjectInfo.SEPARATOR)
    print("OPTIMIZACIJA ZAVRSENA")
    print(ProjectInfo.SEPARATOR)
    print("\nGenerisani fajlovi:")
    print("  -> result.txt                          - Rezultati optimizacije")
    print(ProjectInfo.SEPARATOR)


if __name__ == "__main__":
    main()