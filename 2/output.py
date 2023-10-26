import os
import pprint as pprint

import matplotlib.pyplot as plt
import numpy as np


def gradf(x):
    y1 = -(x[1] * (-2 * x[0] - x[1] + 1)) / 8
    y2 = -(x[0] * (1 - x[0] - 2 * x[1])) / 8
    return np.array([y1, y2])


def better_3d_plot(f, points, filename, show=False):
    # Create a 3D plot
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    # ax.view_init(elev=0, azim=180)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Plot the function surface
    X1, X2 = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    Z = f([X1, X2])
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=1, zorder=1)

    # Plot the points as a scatter plot
    for point in points:
        ax.scatter(point[0], point[1], f(point), 'bo-', s=30, zorder=2)

    # Add an annotation to the last point
    last_point = points[-1]
    last_point_f = f(last_point)

    # Add a text label to the first point with an offset
    first_point = points[0]
    first_point_f = f(first_point)
    offset_x = 0.04
    offset_y = 0.04
    ax.text(first_point[0] + offset_x,
            first_point[1] + offset_y,
            first_point_f,
            "1",
            color='black',
            fontsize=8,
            zorder=2)

    # Add a text label to the last point with an offset
    offset_x = -0.09
    offset_y = -0.09
    ax.text(last_point[0] + offset_x,
            last_point[1] + offset_y,
            last_point_f,
            str(len(points)),
            color='black',
            fontsize=8,
            zorder=2)

    # Set the axis labels and title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')

    # Show the plot
    if not os.path.exists('figures'):
        os.mkdir('figures')
    plt.savefig("figures/" + filename.replace(" ", ""))
    if show:
        plt.show()
    plt.close()


def configurePlot(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.xaxis.get_major_ticks()[0].label1.set_visible(False)


def better_contour_plot(points, filename, show=False):
    fig, ax = plt.subplots()

    X, Y = np.meshgrid(np.arange(0, 1.1, 0.001), np.arange(0, 1.1, 0.001))
    Z = -0.125 * X * Y * (1 - X - Y)

    CS = ax.contour(X, Y, Z, 15, linewidths=0.3)
    ax.clabel(CS, inline=True, fontsize=9)

    x_points = [{
        'val': point[0],
        'num': i + 1
    } for i, point in enumerate(points)]
    y_points = [{
        'val': point[1],
        'num': i + 1
    } for i, point in enumerate(points)]

    x_points_val = [x['val'] for x in x_points]
    y_points_val = [y['val'] for y in y_points]
    ax.plot(x_points_val, y_points_val, 'bo-')

    configurePlot(ax)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    if not os.path.exists('figures'):
        os.mkdir('figures')
    plt.savefig("figures/" + filename.replace(" ", ""))
    if show:
        plt.show()
    plt.close()


def better_draw_triangles(history, filename, present=False, show=False):
    points = [[p[0]['coords'], p[1]['coords'], p[2]['coords'], p[0]['coords']]
              for p in history]

    plt.figure('Nelder-Mead')
    for point in points:
        xs = [x[0] for x in point]
        ys = [y[1] for y in point]
        plt.plot(xs, ys, '-o')

        if present:
            plt.savefig(filename.replace(" ", ""))
            input("Press Enter to continue...")

    if not os.path.exists('figures'):
        os.mkdir('figures')
    plt.savefig("figures/" + filename.replace(" ", ""))
    if show:
        plt.show()
    plt.close()


def print_results(function_uses: int,
                  history: list,
                  res,
                  iterations: int,
                  additional_task_stats: dict = {},
                  stats_val=[]) -> None:
    print(f'Objective function was used {function_uses} times.')
    print(f'Objective function made {iterations} iterations.')
    print('Algorithm\'s results:')
    pprint.pprint(res)

    if 'iterations' and 'function_uses' and 'count' in additional_task_stats:
        print(f'Algorithm used {additional_task_stats["count"]} '
              'additional tasks.')
        print(f'Additional tasks did {additional_task_stats["iterations"]} '
              'iterations in total.')
        print('Additional tasks used the objective function '
              f'{additional_task_stats["function_uses"]} times.')

    print("History:")
    for point in history:
        pprint.pprint(point)
