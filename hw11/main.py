import numpy as np
import matplotlib.pyplot as plt


def solve_electromagnet_optimization():
    N_iter = 1000000

    sigma = 58e6  # provodnost [S/m] (или MS/m * 10^6)
    delta = 1e-2  # pomeraj tacke z0 [m]
    I = 1.0  # struja [A]

    # Slucajne promenljive

    # Poluprecnik a: [0.01, 0.05] m
    a = np.random.uniform(1e-2, 5e-2, N_iter)

    # Duzina b: [0.1, 0.4] m
    b = np.random.uniform(0.1, 0.4, N_iter)

    # Povrsina poprecnog preseka S: [0.5e-6, 3e-6] m^2
    S = np.random.uniform(0.5e-6, 3e-6, N_iter)


    z0 = b / 2.0 + delta  # Pozicija na osi
    term_geo = np.sqrt(np.pi / (4 * S))  # N/2b

    # Racunanje H

    term1_num = z0 + b / 2.0
    term1_den = np.sqrt(a ** 2 + term1_num ** 2)

    term2_num = z0 - b / 2.0
    term2_den = np.sqrt(a ** 2 + term2_num ** 2)

    bracket = (term1_num / term1_den) - (term2_num / term2_den)

    H = (I / 2.0) * term_geo * bracket

    # Racunanje R
    # N -> preko b i s

    R = (2 * np.pi * a * b * term_geo) / (sigma * S)

    data = np.vstack((H, R)).T

    # Sortiranje prema H opadajuce
    sorted_indices = np.argsort(data[:, 0])[::-1]
    sorted_data = data[sorted_indices]

    pareto_H = []
    pareto_R = []

    min_R_so_far = float('inf')

    for i in range(N_iter):
        h_val = sorted_data[i, 0]
        r_val = sorted_data[i, 1]

        if r_val < min_R_so_far:
            pareto_H.append(h_val)
            pareto_R.append(r_val)
            min_R_so_far = r_val

    # Graf
    plt.figure(figsize=(12, 8))

    plt.scatter(R, H, s=1, c='lightgray', label='Sva resenja (Slucajna pretraga)', alpha=0.5)

    # Pareto front
    # plt.plot(pareto_R, pareto_H, c='red', linewidth=2, label='Pareto front')
    plt.scatter(pareto_R, pareto_H, c='red', s=10)  # Tackice na liniji

    plt.xlabel('R [$\Omega$]')
    plt.ylabel('H [A/m]')
    plt.title(f'Pareto front optimizacije elektromagneta\n(Broj iteracijа: $10^6$)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    solve_electromagnet_optimization()
