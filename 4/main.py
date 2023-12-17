import numpy as np

verbose = True


class TableRow:
    def __init__(self, cf: [float], fval: float):
        self.cf = cf
        self.fval = fval

    def __str__(self) -> str:
        return f"{self.cf}, {self.fval}"


def pivoting(index: int, full_table: [TableRow], tolerance=1e-5) -> int:
    min_ratio = float("inf")
    pivot_row_index = -1

    for i, row in enumerate(full_table[1:]):
        if row.cf[index] > tolerance:
            ratio = row.fval / row.cf[index]
            # Bland's Rule: choose the smallest index if there's a tie To avoid cycling, Bland's Rule suggests always
            # choosing the smallest index that satisfies the pivot selection criteria.
            if ratio < min_ratio or (abs(ratio - min_ratio) < tolerance and i < pivot_row_index):
                min_ratio = ratio
                pivot_row_index = i

    if pivot_row_index == -1:
        # Handle the case where no valid pivot is found
        return -1

    return pivot_row_index + 1  # +1 to account for the objective function row


def adjust_table(
        full_table: [TableRow], pivot_row_index: int, pivot_col_index: int
) -> None:
    pivot = full_table[pivot_row_index].cf[pivot_col_index]

    # Divide each element in the pivot row by the pivot.
    # Pivot element should become 1.
    pivot_row = full_table[pivot_row_index]
    pivot_row.fval = pivot_row.fval / pivot
    pivot_row.cf = [x / pivot for x in pivot_row.cf]

    # if verbose:
    #     for row in full_table:
    #         print(row)
    #     print("---------")

    for i in range(len(full_table)):
        if i == pivot_row_index:
            continue

        same_col_as_pivot_el = full_table[i].cf[pivot_col_index]
        new_pivot = full_table[pivot_row_index].cf[pivot_col_index]
        ratio = same_col_as_pivot_el / new_pivot
        if verbose:
            print(f"Ratio: {ratio}")

        full_table[i].cf = [
            x - ratio * pivot_row_el
            for x, pivot_row_el in zip(full_table[i].cf, pivot_row.cf)
        ]

        full_table[i].fval = full_table[i].fval - ratio * pivot_row.fval

    if verbose:
        for row in full_table:
            print(row)


def optimize_linear_program(full_table: [TableRow], var_count: int) -> None:
    function_row = full_table[0]

    while not all(num >= 0 for num in function_row.cf):
        if verbose:
            print("=====================================")
            print("Current table:")
            print_table(full_table)

        pivot_col_index = np.argmin(function_row.cf)
        if verbose:
            print(
                f"Smallest objective function coefficient (column index): {pivot_col_index}"
            )

        pivot_row_index = pivoting(pivot_col_index, full_table)
        if verbose:
            print(f"Smallest non-negative ratio index (row index): {pivot_row_index}")

        pivot = full_table[pivot_row_index].cf[pivot_col_index]
        if verbose:
            print(f"Pivot: {pivot}")

        adjust_table(full_table, pivot_row_index, pivot_col_index)

        if verbose:
            print("=====================================")
            input("Waiting for input. Press any key...\n\n")

    column_sums = [
        sum(row.cf[i] for row in full_table) for i in range(len(full_table[0].cf))
    ]
    base_indexes = [index for index, value in enumerate(column_sums) if value == 1]

    final_vars = [0] * var_count

    for row in full_table[1:]:
        for i in base_indexes:
            if row.cf[i] == 1:
                final_vars[i] = row.fval

    if verbose:
        print("=====================================")
        print("Final Tableau:")
        print_table(full_table)

    print("=====================================")
    print("Final Variables: ")
    print(' | '.join(f"x{i + 1} = {val:8.4f}" for i, val in enumerate(final_vars[:var_count])))

    print(f"\nBasis Indices: {', '.join(f'x{index + 1}' for index in base_indexes)}")
    print(f"Optimum Value: {-function_row.fval:8.4f}")

    return final_vars, [x + 1 for x in base_indexes], -function_row.fval


def print_table(full_table):
    for row in full_table:
        formatted_row = ' | '.join(f"{val:8.2f}" for val in row.cf)
        print(f"[{formatted_row}], {row.fval:8.2f}")


def print_results(res: [[float], [float], float], true_var_count: int):
    print("\nOptimized Results:")
    print(f"Point: {' | '.join(f'{val:8.2f}' for val in res[0][:true_var_count])}")
    print(f"Base: {' | '.join(f'x{index}' for index in res[1])}")
    print(f"Optimum: {res[2]:8.2f}")
    print("=====================================")


def main() -> None:
    # We have 7 variables in total. 4 are given and 3 are slack.
    # Left most element of the list is a constant.
    # First row must be an objective function.
    task = [
        TableRow([2, -3, 0, -5, 0, 0, 0], 0),  # <- Objective function
        TableRow([-1, 1, -1, -1, 1, 0, 0], 8),  # <- Constraints
        TableRow([2, 4, 0, 0, 0, 1, 0], 10),
        TableRow([0, 0, 1, 1, 0, 0, 1], 3),
    ]
    print("=====================================")
    print("Optimizing the generic task:")
    print_results(optimize_linear_program(task, 7), 4)

    task_personal = [
        TableRow([2, -3, 0, -5, 0, 0, 0], 0),
        TableRow([-1, 1, -1, -1, 1, 0, 0], 3),
        TableRow([2, 4, 0, 0, 0, 1, 0], 0),
        TableRow([0, 0, 1, 1, 0, 0, 1], 6),
    ]

    print("\n\n=====================================")
    print("Optimizing the personal task:")
    print_results(optimize_linear_program(task_personal, 7), 4)


if __name__ == "__main__":
    main()
