from typing import Callable

import numpy as np

from nelder_mead import nelder_mead, find_best_points_index

# Funkcija, kurią bandoma minimizuoti
def f(x: list[float]) -> float:
    return -1 * x[0] * x[1] * x[2]


def eq_constraint(x: list[float]) -> float:
    return 2 * (x[0] * x[1] + x[1] * x[2] + x[0] * x[2]) - 1


def ineq_constraint1(x: list[float]) -> float:
    return -1 * x[0]


def ineq_constraint2(x: list[float]) -> float:
    return -1 * x[1]


def ineq_constraint3(x: list[float]) -> float:
    return -1 * x[2]


# Baudos funkcija lygybėms ir nelygybėms
def penalty(
        x: list[float],
        equality_constraints: list[Callable[[list[float]], float]],
        inequality_constraints: list[Callable[[list[float]], float]],
) -> float:
    temp_sum = [0, 0]

    for equality_constraint in equality_constraints:
        temp_sum[0] += equality_constraint(x) ** 2

    for inequality_constraint in inequality_constraints:
        temp_sum[1] += max(0.0, inequality_constraint(x)) ** 2

    return temp_sum[0] + temp_sum[1]


# Padidinta tikslinė funkcija su baudos sąlyga
def b(
        x: list[float],
        r: float,
        equality_constraints: list[Callable[[list[float]], float]],
        inequality_constraint: list[Callable[[list[float]], float]],
):
    return f(x) + (1 / r) * penalty(x, equality_constraints, inequality_constraint)


# Optimizavimo funkcija naudojant Nelder-Mead algoritmą
def optimize(
        starting_point: list[float],
        equality_constraints: list[Callable[[list[float]], float]],
        inequality_constraints: list[Callable[[list[float]], float]],
):
    r = 4
    total_function_calls = 0

    current_point = starting_point
    print("---------------")
    for i in range(1, 100):
        # Susapnuojama padidinta tikslinė funkcija optimizacijai
        b_wrapped = lambda x: b(x, r, equality_constraints, inequality_constraints)

        # Naudojamas Nelder-Mead algoritmas
        simplex, _, function_calls = nelder_mead(b_wrapped, current_point)
        new_point = simplex[find_best_points_index(simplex)]["coords"]
        r = r / 2
        total_function_calls += function_calls

        print(
            f"Iteracija {i}, dabartinis taškas: {new_point}, baudos funkcijos reikšmė: {b_wrapped(new_point)}, r: {r}"
        )

        # Patikrinama, ar pasiekta konvergencija
        if np.linalg.norm(new_point - current_point) <= 0.001:
            current_point = new_point
            break
        current_point = new_point

    return current_point, total_function_calls


def main():
    points = [[0, 0, 0], [1, 1, 1], [3 / 10, 0 / 10, 6 / 10]]
    eqc = [eq_constraint]
    ineqc = [ineq_constraint1, ineq_constraint2, ineq_constraint3]

    for point in points:
        print("----------------------------------")
        print(f"Pradinis taškas: {point}")
        print(f"Baudos funkcija pradiniame taške, kai r = 4: {b(point, 4, eqc, ineqc)}")
        print(f"Funkcijos reikšmė: {f(point)}")

        print("Lygybinių apribojimų reikšmės:")
        print(f"{eq_constraint(point)}")

        print("Nelygybinių apribojimų reikšmės:")
        print(f"{ineq_constraint1(point)}")
        print(f"{ineq_constraint2(point)}")
        print(f"{ineq_constraint3(point)}")

        print("---------------")
        print("Kvadratinės baudos funkcijos reikšmė: ")
        print(f"kai r = 0.002: {b(point, 0.002, eqc, ineqc)}")
        print(f"kai r = 0.04: {b(point, 0.04, eqc, ineqc)}")
        print(f"kai r = 0.2: {b(point, 0.2, eqc, ineqc)}")
        print(f"kai r = 1: {b(point, 1, eqc, ineqc)}")
        print(f"kai r = 5: {b(point, 5, eqc, ineqc)}")
        print(f"kai r = 25: {b(point, 25, eqc, ineqc)}")

        print(
            f"Minimumo taškas ir funkcijos iškvietimų skaičius: {optimize(point, eqc, ineqc)}"
        )


if __name__ == "__main__":
    main()
