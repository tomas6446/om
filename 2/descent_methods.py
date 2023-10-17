import math

import numpy as np


def gradient_descent(f, gradf, start, learning_rate=1, tolerance=0.001):
    steps = [start]  # stat tracing
    steps_val = [f(start)]
    function_uses = 0
    xi = start

    while True:
        gradxi = gradf(xi)  # Find gradient at point X_i
        function_uses += 2

        xi = xi - learning_rate * gradxi  # Find X_i+1 = X_i - gamma * gradf(X_i)
        steps.append(xi)  # stat tracing
        steps_val.append(start)

        if np.linalg.norm(learning_rate * gradxi) < tolerance:
            break
    return steps, xi, function_uses, steps_val


def golden_section(xi, gradxi, func, l=0, r=5, deltax=0.001):
    tau = (-1 + math.sqrt(5)) / 2
    stats = {"function_uses": 0, "iterations": 0}

    # Step 1
    L = r - l
    point1 = r - tau * L
    value_of_function_at_point1 = func(xi + point1 * (-gradxi))
    stats["function_uses"] += 1
    point2 = l + tau * L
    value_of_function_at_point2 = func(xi + point2 * (-gradxi))
    stats["function_uses"] += 1

    while True:
        stats["iterations"] += 1
        # Step 2
        if value_of_function_at_point2 < value_of_function_at_point1:
            l = point1
            L = r - l
            point1 = point2
            value_of_function_at_point1 = value_of_function_at_point2

            point2 = l + tau * L
            value_of_function_at_point2 = func(xi + point2 * (-gradxi))
            stats["function_uses"] += 1
        # Step 3
        else:
            r = point2
            L = r - l
            point2 = point1
            value_of_function_at_point2 = value_of_function_at_point1

            point1 = r - tau * L
            value_of_function_at_point1 = func(xi + point1 * (-gradxi))
            stats["function_uses"] += 1
        # Step 4
        if L < deltax:
            return (point1 + point2) / 2, stats


def steepest_descent(f, gradf, start, tolerance=0.001):
    steps = [start]  # stat tracing
    steps_val = [f(start)]
    stats_additional = {"function_uses": 0, "iterations": 0, "count": 0}
    function_uses = 0
    xi = start

    while True:
        gradxi = gradf(xi)  # Find gradient at point X_i
        function_uses += 2

        learning_rate, stats = golden_section(
            xi, gradxi, f
        )  # Find gamma such: arg min_gamma >= 0 f(X_i - gamma * gradf(X_i))

        stats_additional["function_uses"] += stats["function_uses"]
        stats_additional["iterations"] += stats["iterations"]
        stats_additional["count"] += 1

        xi = xi - learning_rate * gradxi  # Find X_i+1 = X_i - gamma * gradf(X_i)
        steps.append(xi)  # stat tracing
        steps_val.append(start)

        if np.linalg.norm(learning_rate * gradxi) < tolerance:
            break
    return steps, xi, function_uses, stats_additional, steps_val
