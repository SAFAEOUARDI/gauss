import numpy as np
 
 
def back_substitution(U, c):
    """Solve U x = c where U is upper triangular."""
    n = len(c)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x
 
 
def gauss_total_pivoting(A, b, tol=1e-12, verbose=False):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    perm = np.arange(n)      # perm[j] = original index of the unknown in column j
 
    for k in range(n - 1):
        # Pivot search in the whole submatrix
        sub = np.abs(A[k:, k:])
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        i += k
        j += k
        if abs(A[i, j]) < tol:
            raise ValueError("Singular matrix")
 
        if i != k:                               # swap rows
            A[[k, i]] = A[[i, k]]
            b[[k, i]] = b[[i, k]]
        if j != k:                               # swap columns (= swap unknowns)
            A[:, [k, j]] = A[:, [j, k]]
            perm[[k, j]] = perm[[j, k]]
        if verbose and (i != k or j != k):
            print(f"Step {k + 1}: rows {k + 1}<->{i + 1}, columns {k + 1}<->{j + 1}")
 
        # Elimination below the pivot
        for r in range(k + 1, n):
            m = A[r, k] / A[k, k]
            A[r, k:] -= m * A[k, k:]
            b[r] -= m * b[k]
 
        if verbose:
            print(f"After step {k + 1}:\n{np.column_stack((A, b))}\n")
 
    if abs(A[-1, -1]) < tol:
        raise ValueError("Singular matrix")
 
    y = back_substitution(A, b)
    x = np.zeros(n)
    x[perm] = y                                  # put the unknowns back in order
    return x
 
 
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
 
    # Example of the report: exact solution (1, -2, 3)
    A = [[1, 2, 1], [2, 3, 1], [5, 6, 9]]
    b = [0, -1, 20]
    print("Solution:", gauss_total_pivoting(A, b, verbose=True))
    print("numpy   :", np.linalg.solve(A, b))
 
