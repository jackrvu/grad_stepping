import numpy as np
import matplotlib.pyplot as plt
import time

# jack's implementation of a sudoku solver using gradient descent
# mvc final 2025

def read_puzzle(puzzle_str):
    puzzle_str = ''.join(puzzle_str.split())
    grid = []
    for ch in puzzle_str:
        if ch in '123456789':
            grid.append(int(ch))
        elif ch in '.0':
            grid.append(0)
    if len(grid) != 81:
        raise ValueError(f'Puzzle must have 81 cells, found {len(grid)}.')
    return np.array(grid, dtype=int).reshape(9, 9)

    # simple function to read sudoku puzzle from string I input


def print_grid(grid):
    line = '+-------+-------+-------+'
    for r in range(9):
        if r % 3 == 0:
            print(line)
        row = ''
        for c in range(9):
            if c % 3 == 0: row += '| '
            row += str(int(grid[r, c])) + ' '
        print(row + '|')
    print(line)
    # shoutout mr. gao for teaching me how to print grids

def solve_sudoku(initial_clues, 
                 eta0=1.5, max_iter=5000, 
                 w_giv=20., w_row=1., w_col=1., w_sub=1.): # hyperparameters shall work this time
    z = np.zeros((9, 9, 9))
    fixed_mask = np.zeros_like(z, dtype=bool)

    for r in range(9): # initialize logits based on given clues
        for c in range(9):
            if initial_clues[r, c] != 0:
                d = initial_clues[r, c] - 1
                z[r, c] = -10
                z[r, c, d] = 10
                fixed_mask[r, c] = True

    def softmax(z): # softmax
        e = np.exp(z - z.max(axis=2, keepdims=True))
        return e / e.sum(axis=2, keepdims=True)

    loss_history = []
    start_time = time.time()

    for k in range(max_iter):
        p = softmax(z)
        grad = np.zeros_like(z)

        for r in range(9):
            for c in range(9):
                g = initial_clues[r, c]
                if g != 0:
                    pred = (p[r, c] * np.arange(1, 10)).sum()
                    err = 2 * (pred - g)
                    grad[r, c] += w_giv * err * np.arange(1, 10)

        for d in range(9):
            row_sum = p[:, :, d].sum(axis=1, keepdims=True)
            grad[:, :, d] += w_row * 2 * (row_sum - 1)
            col_sum = p[:, :, d].sum(axis=0, keepdims=True)
            grad[:, :, d] += w_col * 2 * (col_sum - 1)
            for br in range(3):
                for bc in range(3):
                    mask = np.zeros((9, 9))
                    mask[3*br:3*br+3, 3*bc:3*bc+3] = 1
                    sub_sum = (p[:, :, d] * mask).sum()
                    grad[:, :, d] += w_sub * 2 * (sub_sum - 1) * mask

        grad *= p * (1 - p)
        grad[fixed_mask] = 0

        eta = eta0 / (1 + 0.01 * k)
        z -= eta * grad

        loss = np.linalg.norm(grad)
        loss_history.append(loss)

        if k % 100 == 0 and loss < 1e-6:
            break

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nGradient descent finished in {k+1} iterations.")
    print(f"Total time elapsed: {elapsed:.3f} seconds")

    # plot loss curve
    plt.plot(loss_history)
    plt.title("Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Gradient Norm)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    final_grid = np.argmax(softmax(z), axis=2) + 1
    return final_grid

if __name__ == "__main__":
    puzzle = """
    53..7....
    6..195...
    .98....6.
    8...6...3
    4..8.3..1
    7...2...6
    .6....28.
    ...419..5
    ....8..79
    """
    clues = read_puzzle(puzzle)
    print("Initial puzzle:")
    print_grid(clues)

    solved = solve_sudoku(clues)
    print("\nSolved:")
    print_grid(solved)
