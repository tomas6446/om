import os

import numpy as np

import descent_methods as dm
import nelder_mead as nm
import output as o


def f(x):
    return -0.125 * x[0] * x[1] * (1 - x[0] - x[1])


def gradf(x):
    y1 = -(x[1] * (-2 * x[0] - x[1] + 1)) / 8
    y2 = -(x[0] * (1 - x[0] - 2 * x[1])) / 8
    return np.array([y1, y2])


def main():
    a = 0
    b = 6
    starting_points = np.array([[0, 0], [1, 1], [a / 10, b / 10]])
    o.better_3d_plot(f, starting_points, "starting_points.png")

    for starting_point in starting_points:
        print("--------------------------------------------------------")
        print(f"Starting point is [{starting_point[0]}, {starting_point[1]}]")

        print("\nGradient descent:")
        history, res, function_uses, stats_val = dm.gradient_descent(
            f, gradf, starting_point)
        o.print_results(function_uses,
                        history,
                        res,
                        len(history) - 1,
                        stats_val=stats_val)
        o.better_3d_plot(f, history,
                         f"gradient_descent_3d_{starting_point}.png")
        o.better_contour_plot(
            history, f"gradient_descent_contour_{starting_point}.png")

        print("\nSteepest descent:")
        history, res, function_uses, stats_additional, stats_val = dm.steepest_descent(
            f, gradf, starting_point)
        o.print_results(function_uses, history, res,
                        len(history) - 1, stats_additional, stats_val)
        o.better_3d_plot(f, history,
                         f"steepest_descent_3d_{starting_point}.png")
        o.better_contour_plot(
            history, f"steepest_descent_contour_{starting_point}.png")

        print("\nNelder-Mead:")
        res, history, function_uses = nm.nelder_mead(f, starting_point)
        o.print_results(function_uses, history, res, len(history))
        o.better_draw_triangles(
            history + [res],
            f"nelder_mead_better_triangles_{starting_point}.png",
            present=False)

        print("--------------------------------------------------------")

        input("Press Enter to continue...")
        os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == "__main__":
    main()
