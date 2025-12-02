import matplotlib.pyplot as plt
import numpy as np


I_L = 7.24                  # 7.24A
I_0 = 4.72 * 10**(-9)       # 4.72nA
a = 1.22                    # 1.22V
R_sh = 144.2                # 144.2om
R_s = 218.5 * 10**(-3)      # 218.5 m_om


class ProjectInfo:
    COURSE = "INZENJERSKI OPTIMIZACIONI ALGORITMI"
    COURSE_EN = "ENGINEERING OPTIMIZATION ALGORITHMS"
    ASSIGNMENT = "Homework Assignment 4"
    AUTHOR = "Jovan Mosurovic"
    INDEX = "2022/0589"
    SEPARATOR = "=" * 65
    SEPARATOR_SMALL = "-" * 65
    PROBLEM = "SOLAR PANEL - OPEN CIRCUIT VOLTAGE CALCULATION"

def print_header():
    print(ProjectInfo.SEPARATOR)
    print(ProjectInfo.COURSE)
    print(f"({ProjectInfo.COURSE_EN})")
    print(ProjectInfo.ASSIGNMENT)
    print(ProjectInfo.SEPARATOR_SMALL)
    print(f"Author: {ProjectInfo.AUTHOR} {ProjectInfo.INDEX}")
    print(ProjectInfo.SEPARATOR)
    print()
    print(ProjectInfo.PROBLEM)
    print(ProjectInfo.SEPARATOR)


def f(U):
    return -I_L + I_0 * (np.exp(U / a) - 1) + U / R_sh

# first derivative of f(U)
def df(U):
    return I_0 / a * np.exp(U / a) + 1 / R_sh

def make_plot():
    fig = plt.figure(figsize=(10, 6))

    # For generating 250 dots
    U = np.linspace(0, 30, 250)

    # Calculate the current
    I = f(U)

    plt.plot(U, I, 'b-', linewidth=2)
    plt.xlabel('Napon(Voltage) U [V]', fontsize=12)
    plt.ylabel('Struja(Current) I [A]', fontsize=12)
    plt.title('Graph for function f(U)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def newton_method(x0, tolerance=1e-6, max_iterations=1000):
    """
    Newtong's method for finding zero's in functions

    Iterative formula: x_{k+1} = x_k - f(x_k) / f'(x_k)

    x0: starting value
    tolerance: (±1 µV = 1e-6 V)
    """

    x = x0

    print(f"{'Iteration':<12} {'x_k [V]':<20} {'f(x_k) [A]':<20} {'|Δx| [V]':<15}")
    print(ProjectInfo.SEPARATOR_SMALL)

    for i in range(max_iterations):
        fx = f(x)
        dfx = df(x)

        # Newton's iteration: x_{k+1} = x_k - f(x_k) / f'(x_k)
        x_new = x - fx / dfx

        delta_x = abs(x_new - x)

        print(f"{i:<12} {x:<20.10f} {fx:<20.2e} {delta_x:<15.2e}")

        if delta_x < tolerance:
            print(f"\nConvergence achieved after {i + 1} iterations")
            print(f"Solution: U = {x_new:.10f} V")
            print(f"Check: f(U) = {f(x_new):.2e} A")
            return x_new

        x = x_new

    print(f"\nMaximum number of iterations reached ({max_iterations})")
    return x


def save_result(U_zero):
    with open('result.txt', 'w', encoding='utf-8') as file:
        file.write(ProjectInfo.SEPARATOR + "\n")
        file.write(ProjectInfo.COURSE + "\n")
        file.write(f"({ProjectInfo.COURSE_EN})\n")
        file.write(ProjectInfo.ASSIGNMENT + "\n")
        file.write(f"Author: {ProjectInfo.AUTHOR} {ProjectInfo.INDEX}\n")
        file.write(ProjectInfo.SEPARATOR + "\n\n")

        file.write("OPTIMAL SOLAR PANEL OPEN-CIRCUIT VOLTAGE\n")
        file.write(ProjectInfo.SEPARATOR + "\n\n")

        file.write("Method: Newton's method (Newton-Raphson)\n\n")

        file.write("Parameters:\n")
        file.write(ProjectInfo.SEPARATOR_SMALL + "\n")
        file.write(f"I_L   = {I_L} A (light-generated current)\n")
        file.write(f"I_0   = {I_0:.2e} A (reverse saturation current)\n")
        file.write(f"a     = {a} V (modified ideality parameter)\n")
        file.write(f"R_sh  = {R_sh} Ω (shunt resistance)\n")
        file.write(f"R_s   = {R_s * 1e3} mΩ (series resistance)\n\n")

        file.write("Result:\n")
        file.write(ProjectInfo.SEPARATOR_SMALL + "\n")
        file.write(f"U = {U_zero:.10f} V\n")
        file.write(f"U = {U_zero * 1e6:.4f} µV\n\n")

        file.write("Verification:\n")
        file.write(ProjectInfo.SEPARATOR_SMALL + "\n")
        file.write(f"f(U) = {f(U_zero):.2e} A\n\n")

        file.write(f"Accuracy: ±1 µV (±1e-6 V)\n")
        file.write(ProjectInfo.SEPARATOR + "\n")

    print(f"Result saved to 'result.txt'")

def main():
    print_header()

    print("\nProblem configuration:")
    print(f"Objective: Find voltage U where f(U) = 0")
    print(f"Method: Newton's method")
    print(f"Required accuracy: ±1 µV")
    print(f"Voltage range: [0, 30] V")

    print(f"\n{ProjectInfo.SEPARATOR}")
    print("[1] Plotting the graph of f(U)...")
    print(ProjectInfo.SEPARATOR)
    make_plot()

    print(f"\n{ProjectInfo.SEPARATOR}")
    print("[2] Finding the zero of the function using Newton's method...")
    print(ProjectInfo.SEPARATOR)

    # Initial value (from the graph, the zero is around 21-22 V)
    x0 = 20.0

    tolerance = 1e-6

    U_zero = newton_method(x0, tolerance)

    print(f"\n{ProjectInfo.SEPARATOR}")
    print("[3] Saving the result...")
    print(ProjectInfo.SEPARATOR)
    save_result(U_zero)

    print(f"\n{ProjectInfo.SEPARATOR}")
    print("COMPLETED")
    print(ProjectInfo.SEPARATOR)
    print("\nGenerated files:")
    print("-> graph_f_U.png  - Graph of f(U)")
    print("-> result.txt     - Numerical solution with ±1 µV accuracy")
    print(ProjectInfo.SEPARATOR)

if __name__ == "__main__":
    main()