import numpy as np
from scipy.optimize import linprog



class ProjectInfo:
    COURSE = "INZENJERSKI OPTIMIZACIONI ALGORITMI"
    COURSE_EN = "ENGINEERING OPTIMIZATION ALGORITHMS"
    ASSIGNMENT = "Homework Assignment 6"
    AUTHOR = "Jovan Mosurovic"
    INDEX = "2022/0589"
    SEPARATOR = "=" * 65
    SEPARATOR_SMALL = "-" * 65
    PROBLEM = "TRANSPORT PROBLEM - LINEAR PROGRAMMING"


def print_header():
    print()
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
    print()

def solve():
    transport_costs = np.array([
        [2.0, 3.2, 2.4],
        [0.0, 1.6, 4.8],
        [1.6, 0.0, 2.8],
        [2.8, 0.8, 2.0],
        [4.8, 2.8, 0.0]
    ])

    production = np.array([700, 500, 100, 800, 400])
    center_capacity = np.array([900, 900, 900])

    print("Cene transporta: ")
    print("          Centar 2  Centar 3  Centar 5")
    for i in range(5):
        print(f"Fabrika {i + 1}:   {transport_costs[i, 0]:.1f}      {transport_costs[i, 1]:.1f}      {transport_costs[i, 2]:.1f}")

    print(f"\nProdukcija: {production}")
    print(f"Kapaciteti: {center_capacity}\n")

    num_factories = 5
    num_centers = 3
    num_variables = num_factories * num_centers

    c = transport_costs.flatten()

    # Ogranicenja: kapacitet centara (nejednakosti)
    A_ub = []
    b_ub = []
    for j in range(num_centers):
        constraint = np.zeros(num_variables)
        for i in range(num_factories):
            constraint[i * num_centers + j] = 1
        A_ub.append(constraint)
        b_ub.append(center_capacity[j])

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Ogranicenja: produkcija fabrika (jednakosti)
    A_eq = []
    b_eq = []
    for i in range(num_factories):
        constraint = np.zeros(num_variables)
        for j in range(num_centers):
            constraint[i * num_centers + j] = 1
        A_eq.append(constraint)
        b_eq.append(production[i])

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    bounds = [(0, None) for _ in range(num_variables)]

    print(ProjectInfo.SEPARATOR_SMALL)
    print("Optimizacija: HiGHS Dual Simplex\n")

    # Resavanje
    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs-ds"
    )

    if not result.success:
        print("Optimizacija neuspesna!")
        return None, None

    print("Optimalno resenje pronadjeno")

    x_optimal = result.x
    total_cost = result.fun
    solution_matrix = x_optimal.reshape(5, 3)

    # Prikaz rešenja
    print(ProjectInfo.SEPARATOR_SMALL)
    print("OPTIMALNI PLAN TRANSPORTA [broj racunara]\n")
    print("            Centar 2  Centar 3  Centar 5  |  Ukupno")
    print(ProjectInfo.SEPARATOR_SMALL)

    for i in range(5):
        row_sum = np.sum(solution_matrix[i, :])
        print(
            f"Fabrika {i + 1}:   {solution_matrix[i, 0]:6.0f}    {solution_matrix[i, 1]:6.0f}    {solution_matrix[i, 2]:6.0f}  |  {row_sum:6.0f}")

    print(ProjectInfo.SEPARATOR_SMALL)
    col_sums = np.sum(solution_matrix, axis=0)
    print(f"Ukupno:      {col_sums[0]:6.0f}    {col_sums[1]:6.0f}    {col_sums[2]:6.0f}  |  {np.sum(col_sums):6.0f}")

    print(f"\n{ProjectInfo.SEPARATOR_SMALL}")
    print(f"MINIMALNA CENA TRANSPORTA: {total_cost:.2f} EUR")
    print(ProjectInfo.SEPARATOR_SMALL + "\n")

    return x_optimal, total_cost, solution_matrix, transport_costs


def save_results(x_optimal, total_cost, solution_matrix, transport_costs):
    if x_optimal is None:
        return

    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(ProjectInfo.SEPARATOR + "\n")
        f.write(ProjectInfo.COURSE + "\n")
        f.write(f"({ProjectInfo.COURSE_EN})\n")
        f.write(ProjectInfo.ASSIGNMENT + "\n")
        f.write(f"Author: {ProjectInfo.AUTHOR} ({ProjectInfo.INDEX})\n")
        f.write(ProjectInfo.SEPARATOR + "\n\n")

        f.write("OPTIMALNO RESENJE\n")
        f.write(ProjectInfo.SEPARATOR_SMALL + "\n\n")

        f.write("Vektor resenja:\n")
        f.write("x = [")
        for i, val in enumerate(x_optimal):
            if i > 0:
                f.write(", ")
            f.write(f"{val:.1f}")
        f.write("]\n\n")

        f.write("Matrica transporta [broj racunara]:\n\n")
        f.write("            Centar 2  Centar 3  Centar 5\n")
        for i in range(5):
            f.write(f"Fabrika {i + 1}:   {solution_matrix[i, 0]:6.0f}    {solution_matrix[i, 1]:6.0f}    {solution_matrix[i, 2]:6.0f}\n")

        f.write("\nDetaljan plan:\n")
        for i in range(5):
            for j in range(3):
                computers = solution_matrix[i, j]
                if computers > 0:
                    center_name = [2, 3, 5][j]
                    cost = computers * transport_costs[i, j]
                    f.write(f"F{i + 1} -> C{center_name}: {computers:6.0f} x {transport_costs[i, j]:.1f} = {cost:.2f} EUR\n")

        f.write(f"\nMinimalna ukupna cena: {total_cost:.2f} EUR\n")
        f.write(ProjectInfo.SEPARATOR + "\n")

def main():
    print_header()
    x_optimal, total_cost, solution_matrix, transport_costs = solve()
    save_results(x_optimal, total_cost, solution_matrix, transport_costs)

    print(ProjectInfo.SEPARATOR)
    print("OPTIMIZACIJA ZAVRSENA")
    print(ProjectInfo.SEPARATOR)
    print("\nGenerisani fajlovi:")
    print("-> result.txt     - Fajl sa rezultatima")
    print(ProjectInfo.SEPARATOR + "\n")

if __name__ == "__main__":
    main()