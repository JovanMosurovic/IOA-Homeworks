import numpy as np
import random


class ProjectInfo:
    COURSE = "INZENJERSKI OPTIMIZACIONI ALGORITMI"
    COURSE_EN = "ENGINEERING OPTIMIZATION ALGORITHMS"
    ASSIGNMENT = "Homework Assignment 10"
    AUTHOR = "Jovan Mosurovic"
    INDEX = "2022/0589"
    SEPARATOR = "=" * 80
    SEPARATOR_SMALL = "-" * 80
    PROBLEM = "STEINER TREE PROBLEM - PARTICLE SWARM OPTIMIZATION"


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


# Definisanje fiksnih tačaka
A = np.array([1, 4, 0])
B = np.array([3, 2, 0])
C = np.array([2, 7, 1])
D = np.array([6, 3, 3])
E = np.array([7, 6, 5])
F = np.array([5, 7, 4])


def calculate_distance(p1, p2):
    """Izracunava Euklidsko rastojanje izmedju dve tacke"""
    return np.linalg.norm(p1 - p2)


def load_data():
    print("\n[1] UCITAVANJE PODATAKA")
    print(ProjectInfo.SEPARATOR_SMALL)
    print(f"Broj fiksnih tacaka: 6")
    print(f"Broj Steiner tacaka (nepoznatih): 2")
    print(f"Ukupno promenljivih: 6 (x1, y1, z1, x2, y2, z2)")
    print(f"\nFiksne tacke:")
    print(f"  A = ({A[0]}, {A[1]}, {A[2]})")
    print(f"  B = ({B[0]}, {B[1]}, {B[2]})")
    print(f"  C = ({C[0]}, {C[1]}, {C[2]})")
    print(f"  D = ({D[0]}, {D[1]}, {D[2]})")
    print(f"  E = ({E[0]}, {E[1]}, {E[2]})")
    print(f"  F = ({F[0]}, {F[1]}, {F[2]})")
    print(f"\nTopologija:")
    print(f"  A, B, C -> S1 -> S2 <- D, E, F")


def objective_function(solution):
    """
    Funkcija cilja - ukupna duzina puta
    solution = [x1, y1, z1, x2, y2, z2]
    """
    S1 = np.array(solution[:3])
    S2 = np.array(solution[3:])

    # Rastojanja od A, B, C do S1
    dist_AS1 = calculate_distance(A, S1)
    dist_BS1 = calculate_distance(B, S1)
    dist_CS1 = calculate_distance(C, S1)

    # Rastojanja od D, E, F do S2
    dist_DS2 = calculate_distance(D, S2)
    dist_ES2 = calculate_distance(E, S2)
    dist_FS2 = calculate_distance(F, S2)

    # Rastojanje izmedju S1 i S2
    dist_S1S2 = calculate_distance(S1, S2)

    # Ukupna duzina
    total_distance = (dist_AS1 + dist_BS1 + dist_CS1 +
                      dist_DS2 + dist_ES2 + dist_FS2 + dist_S1S2)

    return total_distance


def pso_optimization(n_particles=50, n_iterations=1000, w=0.7, c1=1.5, c2=1.5):
    """
    Particle Swarm Optimization algoritam

    Parametri:
    - n_particles: broj cestica u jatu
    - n_iterations: broj iteracija
    - w: inercijski faktor
    - c1: kognitivni faktor
    - c2: socijalni faktor
    """

    # Postavljanje seed-a za reproduktivnost
    np.random.seed(42)
    random.seed(42)

    # Dimenzionalnost problema (6 - x1,y1,z1,x2,y2,z2)
    dim = 6

    # Granice prostora pretrage
    lower_bound = np.array([0, 0, 0, 4, 2, 2])
    upper_bound = np.array([5, 8, 3, 8, 8, 6])

    # Inicijalizacija pozicija i brzina čestica
    particles_position = np.random.uniform(lower_bound, upper_bound, (n_particles, dim))
    particles_velocity = np.random.uniform(-1, 1, (n_particles, dim))

    # Inicijalizacija najboljih pozicija
    personal_best_position = particles_position.copy()
    personal_best_value = np.array([objective_function(p) for p in particles_position])

    # Globalno najbolje rešenje
    global_best_idx = np.argmin(personal_best_value)
    global_best_position = personal_best_position[global_best_idx].copy()
    global_best_value = personal_best_value[global_best_idx]

    # PSO iteracije
    for iteration in range(n_iterations):
        for i in range(n_particles):
            # Ažuriranje brzine
            r1 = random.random()
            r2 = random.random()

            cognitive_component = c1 * r1 * (personal_best_position[i] - particles_position[i])
            social_component = c2 * r2 * (global_best_position - particles_position[i])

            particles_velocity[i] = (w * particles_velocity[i] +
                                     cognitive_component +
                                     social_component)

            # Ažuriranje pozicije
            particles_position[i] = particles_position[i] + particles_velocity[i]

            # Ograničenje pozicije na dozvoljeni prostor
            particles_position[i] = np.clip(particles_position[i], lower_bound, upper_bound)

            # Evaluacija nove pozicije
            current_value = objective_function(particles_position[i])

            # Ažuriranje lične najbolje pozicije
            if current_value < personal_best_value[i]:
                personal_best_value[i] = current_value
                personal_best_position[i] = particles_position[i].copy()

                # Ažuriranje globalne najbolje pozicije
                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = particles_position[i].copy()

    return global_best_position, global_best_value


def analyze_solution(solution, fitness):
    print(f"\n[3] REZULTATI OPTIMIZACIJE")
    print(ProjectInfo.SEPARATOR_SMALL)

    S1 = solution[:3]
    S2 = solution[3:]

    print("\nPronadjene Steiner tacke (precizne vrednosti):")
    print(f"    S1: x1 = {S1[0]:.15f}, y1 = {S1[1]:.15f}, z1 = {S1[2]:.15f}")
    print(f"    S2: x2 = {S2[0]:.15f}, y2 = {S2[1]:.15f}, z2 = {S2[2]:.15f}")

    print("\nPronadjene Steiner tacke (zaokruzene vrednosti):")
    print(f"    S1({S1[0]:.3f}, {S1[1]:.3f}, {S1[2]:.3f})")
    print(f"    S2({S2[0]:.3f}, {S2[1]:.3f}, {S2[2]:.3f})")

    print(f"\nMinimalna duzina puta (D): {fitness:.15f}")


def save_solution(solution, fitness):
    print(f"\n[4] CUVANJE REZULTATA")
    print(ProjectInfo.SEPARATOR_SMALL)

    S1 = solution[:3]
    S2 = solution[3:]

    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(ProjectInfo.SEPARATOR + "\n")
        f.write(ProjectInfo.COURSE + "\n")
        f.write(f"({ProjectInfo.COURSE_EN})\n")
        f.write(ProjectInfo.ASSIGNMENT + "\n")
        f.write(f"Autor: {ProjectInfo.AUTHOR} ({ProjectInfo.INDEX})\n")
        f.write(ProjectInfo.SEPARATOR + "\n\n")

        f.write(f"S1({S1[0]:.3f}, {S1[1]:.3f}, {S1[2]:.3f})\n")
        f.write(f"S2({S2[0]:.3f}, {S2[1]:.3f}, {S2[2]:.3f})\n")
        f.write(f"Distance: {fitness}\n\n")

        f.write("Precizne vrednosti Steiner tacaka:\n")
        f.write(f"    S1: x1 = {S1[0]:.15f}, y1 = {S1[1]:.15f}, z1 = {S1[2]:.15f}\n")
        f.write(f"    S2: x2 = {S2[0]:.15f}, y2 = {S2[1]:.15f}, z2 = {S2[2]:.15f}\n\n")

        f.write(ProjectInfo.SEPARATOR + "\n")

    print("Rezultati sacuvani u: result.txt")


def main():
    print_header()

    load_data()

    print(f"\n[2] PARAMETRI ALGORITMA")
    print(ProjectInfo.SEPARATOR_SMALL)
    n_particles = 50
    n_iterations = 1000
    print(f"Velicina jata (broj cestica): {n_particles}")
    print(f"Maksimalan broj iteracija: {n_iterations}")
    print(f"Inercijski faktor w: 0.7")
    print(f"Kognitivni faktor c1: 1.5")
    print(f"Socijalni faktor c2: 1.5")

    # Pokretanje PSO optimizacije
    best_solution, best_fitness = pso_optimization(
        n_particles=n_particles,
        n_iterations=n_iterations,
        w=0.7,
        c1=1.5,
        c2=1.5
    )

    analyze_solution(best_solution, best_fitness)

    save_solution(best_solution, best_fitness)

    print()
    print(ProjectInfo.SEPARATOR)
    print("OPTIMIZACIJA ZAVRSENA")
    print(ProjectInfo.SEPARATOR)


if __name__ == "__main__":
    main()