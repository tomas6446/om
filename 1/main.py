import math
import matplotlib.pyplot as plt
import numpy as np


def funw(x):
    return (x ** 2) ** 2 / 6 - 1


def f(x, stats):
    stats['call_count'] += 1
    return funw(x)


def fder(x, stats):
    stats['call_count'] += 1
    return ((2 * x) ** 3) / 3


def fsecder(x, stats):
    stats['call_count'] += 1
    return (2 * x) ** 2


def bisection(l, r, deltax):
    stats = {'steps': 0, 'call_count': 0, 'points': [], 'interval': []}

    # Step 1
    xm = (l + r) / 2
    L = r - l
    fxm = f(xm, stats)
    stats['points'].append(xm)  # Save a point
    stats['interval'].append(L)  # Save interval

    while True:
        stats['steps'] += 1
        # Step 2
        x1 = l + L / 4
        fx1 = f(x1, stats)
        x2 = r - L / 4
        fx2 = f(x2, stats)

        # Step 3
        if fx1 < fxm:
            r = xm
            xm = x1
            fxm = fx1
        # Step 4
        elif fx2 < fxm:
            l = xm
            xm = x2
            fxm = fx2
        # Step 5
        else:
            l = x1
            r = x2

        stats['points'].append(xm)  # Save a point
        stats['interval'].append(L)  # Save interval

        # Step 6
        L = r - l
        if L < deltax:
            return (xm, stats, funw(xm))


def golden_section(l, r, deltax):
    stats = {
        'steps': 0,
        'call_count': 0,
        "intervals": [],
        'points': [],
        'interval': []
    }

    # Step 1
    t = (-1 + math.sqrt(5)) / 2
    L = r - l
    x1 = r - t * L
    x2 = l + t * L
    fx1 = f(x1, stats)
    fx2 = f(x2, stats)
    stats['points'].append((x1 + x2) / 2)  # Save a point
    stats['intervals'].append((l, r))
    stats['interval'].append(L)  # Save interval

    while True:
        stats['steps'] += 1
        # Step 2
        if fx2 < fx1:
            l = x1
            L = r - l
            x1 = x2
            fx1 = fx2

            x2 = l + t * L
            fx2 = f(x2, stats)
        # Step 3
        else:
            r = x2
            L = r - l
            x2 = x1
            fx2 = fx1

            x1 = r - t * L
            fx1 = f(x1, stats)

        stats['points'].append((x1 + x2) / 2)  # Save a point
        stats['intervals'].append((l, r))
        stats['interval'].append(L)  # Save interval

        # Step 4
        if L < deltax:
            return (x1 + x2) / 2, stats, funw((x1 + x2) / 2)


def newtons(x0, deltax):
    stats = {'steps': 0, 'call_count': 0, 'points': [x0], 'interval': []}

    xinext = x0
    while True:
        stats['steps'] += 1
        xi = xinext
        xinext = xi - (fder(xi, stats) / fsecder(xi, stats))

        stats['points'].append(xinext)  # Save a point
        stats['interval'].append(abs(xi - xinext))  # Save interval

        if abs(xi - xinext) < deltax:
            return xinext, stats, funw(xinext)


def generate_graph(stats, filename: str) -> None:
    x = np.linspace(0, 10, 10000)
    y = funw(x)
    points = np.array(stats['points'])

    # Set color and size
    sizes = np.random.uniform(15, 80, len(points))
    colors = np.random.uniform(15, 80, len(points))

    # Find y from solutions
    pointsy = []
    for point in points:
        pointsy.append(funw(point))

    fig, ax = plt.subplots()

    ax.plot(x, y, 'r', zorder=1)
    ax.grid(alpha=.6, linestyle='--')
    ax.scatter(points, pointsy, s=sizes, c=colors, vmin=0, vmax=100)

    texts = []
    for i, txt in enumerate(points):
        bbox_props = dict(boxstyle="round,pad=0.5",
                          fc="white",
                          ec="gray",
                          lw=1)
        text = ax.annotate(str(i + 1), (points[i], pointsy[i]),
                           xytext=(-5, 15),
                           textcoords='offset points',
                           bbox=bbox_props)

    plt.axhline(0, color='black', alpha=0.6)
    plt.axvline(0, color='black', alpha=0.6)
    plt.xlim((-0.5, 5.2))
    plt.ylim((-1.5, 42))
    plt.savefig(filename)
    # plt.show()


def print_results(algo_name: str, res):
    print("-------------------")
    print(algo_name)
    print("-------------------")

    print("xmin = ", res[0])
    print("ymin = ", res[2])
    print("steps = ", res[1]['steps'])
    print("call_count = ", res[1]['call_count'])

    print("-------------------")
    print("Algorithm used these points:")
    if algo_name == "Golden-section search: ":
        for point, interval, intervals in zip(res[1]['points'],
                                              res[1]['interval'],
                                              res[1]['intervals']):
            print(
                f'Point: {point:.20f}, interval: {interval:.20f}, l: {intervals[0]:.20f}, r: {intervals[1]:.20f}'
            )
    else:
        for point, interval in zip(res[1]['points'], res[1]['interval']):
            print(f'Point: {point:.20f}, interval: {interval:.20f}')


def main():
    # Studento numerio sk.: a = 0; b = 6
    # My function: f(x) = (x^2)^2 / 6) - 1
    # Derivative: f'(x) = (2x^3) / 3
    # Second derivative: f''(x) = 2x^2
    x0 = 0
    xn = 10
    deltax = 0.0001
    newtons_start = 5

    bires = bisection(x0, xn, deltax)
    golres = golden_section(x0, xn, deltax)
    newres = newtons(newtons_start, deltax)

    print_results("Bisection method: ", bires)
    print_results("Golden-section search: ", golres)
    print_results("Newton's method:", newres)

    generate_graph(bires[1], "bisection.png")
    generate_graph(golres[1], "golden-section.png")
    generate_graph(newres[1], "newtons.png")


if __name__ == "__main__":
    main()
