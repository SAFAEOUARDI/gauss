import numpy as np
 
 
def back_substitution(U, c):
    """Solve U x = c where U is upper triangular."""
    n = len(c)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x
 
 
def gauss_partial_pivoting(A, b, tol=1e-12, verbose=False):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
 
    for k in range(n - 1):
        # Pivot search in column k, rows k..n-1
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if abs(A[p, k]) < tol:
            raise ValueError("Singular matrix")
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
            if verbose:
                print(f"Step {k + 1}: swap rows {k + 1} and {p + 1}")
 
        # Elimination below the pivot
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]
 
        if verbose:
            print(f"After step {k + 1}:\n{np.column_stack((A, b))}\n")
 
    if abs(A[-1, -1]) < tol:
        raise ValueError("Singular matrix")
    return back_substitution(A, b)
 
 
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
 
    # Example of the report: exact solution (1, -2, 3)
    A = [[1, 2, 1], [2, 3, 1], [5, 6, 9]]
    b = [0, -1, 20]
    print("Solution:", gauss_partial_pivoting(A, b, verbose=True))
    print("numpy   :", np.linalg.solve(A, b))
