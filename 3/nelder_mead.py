import numpy as np
import math
import copy


def generate_points(
    f, starting_point, alpha=0.5
):  # Alpha is basically the length of the side of the initial simplex
    x0 = np.array(starting_point)
    n = len(x0)
    X = [x0]
    sqrt2 = math.sqrt(2)

    for i in range(n):
        si = []
        for j in range(n):
            if i == j:
                si.append((math.sqrt(n + 1) - 1) / (n * sqrt2) * alpha)
            else:
                si.append((math.sqrt(n + 1) + n - 1) / (n * sqrt2) * alpha)
        X.append(x0 + si)

    # Form a list of dictionaries with point coordinates and the value of a function
    return [
        {"coords": np.array(x[0]), "value": x[1]} for x in zip(X, [f(x) for x in X])
    ]


def find_centroid(f, points, n, worst_points_index):
    centroid_coords = (
        1
        / n
        * sum([v["coords"] for i, v in enumerate(points) if i != worst_points_index])
    )
    return {"coords": centroid_coords, "value": f(centroid_coords)}


def step(f, points, centroid, worst_points_index, alpha):
    new_point_coords = centroid["coords"] + alpha * (
        centroid["coords"] - points[worst_points_index]["coords"]
    )
    return {"coords": new_point_coords, "value": f(new_point_coords)}


def find_worst_points_index(points):
    return np.array([point["value"] for point in points]).argmax()


def find_second_worst_points_index(points):
    worst_points_index = find_worst_points_index
    return np.array(
        [v["value"] for i, v in enumerate(points) if i != worst_points_index]
    ).argmax()


def find_best_points_index(points):
    return np.array([point["value"] for point in points]).argmin()


def shrink(f, points, gamma=0.5):
    X = []

    for i, x in enumerate(points):
        if i == find_best_points_index(points):
            X.append(x)
            continue

        best_point = points[find_best_points_index(points)]
        x_coords = best_point["coords"] + gamma * (x["coords"] - best_point["coords"])
        X.append({"coords": x_coords, "value": f(x_coords)})

    return X


def nelder_mead(f, starting_point, tolerance=0.001):
    # Stat tracing
    triangles = []
    function_calls = 0

    # Generate Simplex Points
    simplex = generate_points(f, starting_point)
    function_calls += len(simplex)

    n = len(simplex) - 1  # Number of variables

    for i in range(1, 50):
        # Select Worst Point
        worst_points_index = find_worst_points_index(simplex)

        # Find Centroid
        centroid = find_centroid(f, simplex, n, worst_points_index)
        function_calls += 1

        # Reflection
        xr = step(f, simplex, centroid, worst_points_index, 1)
        function_calls += 1

        triangles.append(copy.deepcopy(simplex))  # Stats

        # Try Expansion
        if (
            xr["value"] <= simplex[find_best_points_index(simplex)]["value"]
        ):  # F(x_r) <= F(x^(0))
            xe = step(f, simplex, centroid, worst_points_index, 2)
            function_calls += 1

            if (
                xe["value"] <= simplex[find_best_points_index(simplex)]["value"]
            ):  # F(x_e) <= F(x^(0))
                simplex[worst_points_index] = xe
            else:
                simplex[worst_points_index] = xr
        # Reflected is fine
        elif xr["value"] <= simplex[find_second_worst_points_index(simplex)]["value"]:
            simplex[worst_points_index] = xr
        # Inside contraction
        elif xr["value"] >= simplex[worst_points_index]["value"]:
            xic = step(f, simplex, centroid, worst_points_index, -0.5)
            function_calls += 1

            if xic["value"] <= simplex[worst_points_index]["value"]:
                simplex[worst_points_index] = xic
            # Shrink
            else:
                simplex = shrink(f, simplex)
                function_calls += n
        # Outside contraction
        else:
            xoc = step(f, simplex, centroid, worst_points_index, 0.5)
            function_calls += 1

            if xoc["value"] <= simplex[worst_points_index]["value"]:
                simplex[worst_points_index] = xoc
            # Shrink
            else:
                simplex = shrink(f, simplex)
                function_calls += n

        if (
            np.linalg.norm(
                simplex[find_worst_points_index(simplex)]["coords"]
                - simplex[find_best_points_index(simplex)]["coords"]
            )
            <= tolerance
        ):
            break

    return simplex, triangles, function_calls
