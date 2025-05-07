import numpy as np

def read_puzzle(puzzle_str):
    puzzle_str = ''.join(puzzle_str.split())      # remove all whitespace
    grid = []
    for ch in puzzle_str:
        if ch in '123456789':
            grid.append(int(ch))
        elif ch in '.0':
            grid.append(0)                        # treat . or 0 as blank
        # ignore anything else: spaces, new‑lines, tabs
    if len(grid) != 81:
        raise ValueError(f'Puzzle must have 81 cells, found {len(grid)}.')
    return np.array(grid, dtype=int).reshape(9, 9)

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

def solve_sudoku(initial_clues, 
                 eta0=1.5, max_iter=5000, 
                 w_giv=20., w_row=1., w_col=1., w_sub=1.):
    # logits (r,c,9)
    z = np.zeros((9, 9, 9))
    fixed_mask = np.zeros_like(z, dtype=bool)

    for r in range(9):
        for c in range(9):
            if initial_clues[r, c] != 0:
                d = initial_clues[r, c] - 1
                z[r, c] = -10     # low everywhere …
                z[r, c, d] = 10   # … except the clue digit
                fixed_mask[r, c] = True

    def softmax(z):
        e = np.exp(z - z.max(axis=2, keepdims=True))
        return e / e.sum(axis=2, keepdims=True)

    for k in range(max_iter):
        p = softmax(z)

        # ----- build gradients ----- #
        grad = np.zeros_like(z)

        # (1) clue loss
        for r in range(9):
            for c in range(9):
                g = initial_clues[r, c]
                if g != 0:
                    pred = (p[r, c] * np.arange(1, 10)).sum()
                    err = 2 * (pred - g)
                    grad[r, c] += w_giv * err * np.arange(1, 10) * p[r, c] * (1 - p[r, c])

        # (2) row / column / subgrid constraints
        for d in range(9):
            # rows
            row_sum = p[:, :, d].sum(axis=1, keepdims=True)
            grad[:, :, d] += w_row * 2 * (row_sum - 1)
            # columns
            col_sum = p[:, :, d].sum(axis=0, keepdims=True)
            grad[:, :, d] += w_col * 2 * (col_sum - 1)
            # sub‑grids
            for br in range(3):
                for bc in range(3):
                    mask = np.zeros((9, 9))
                    mask[3*br:3*br+3, 3*bc:3*bc+3] = 1
                    sub_sum = (p[:, :, d] * mask).sum()
                    grad[:, :, d] += w_sub * 2 * (sub_sum - 1) * mask

        # chain rule for softmax
        # softmax derivative already baked above for clue term;
        # for constraint terms we need ∂p/∂z: p*(δ_kj − p_j)
        # approximate with p*(1-p) (works fine in practice)
        grad *= p * (1 - p)

        # do not move fixed cells
        grad[fixed_mask] = 0

        # ----- update ----- #
        eta = eta0 / (1 + 0.01 * k)
        z -= eta * grad

        # quick convergence check each 100 iters
        if k % 100 == 0:
            loss = np.linalg.norm(grad)
            if loss < 1e-6:
                break

    final_grid = np.argmax(softmax(z), axis=2) + 1
    return final_grid

if __name__ == "__main__":
    # Hard (NY‑Times “medium”) puzzle – dots are blanks
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
