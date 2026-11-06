import numpy as np
 
 
def lu_doolittle(A, tol=1e-12):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):                    # row i of U
            U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]
        if abs(U[i, i]) < tol:
            raise ValueError("Zero pivot: LU decomposition does not exist")
        for j in range(i + 1, n):                # column i of L
            L[j, i] = (A[j, i] - L[j, :i] @ U[:i, i]) / U[i, i]
    return L, U
 
 
def lu_crout(A, tol=1e-12):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.eye(n)
    for j in range(n):
        for i in range(j, n):                    # column j of L
            L[i, j] = A[i, j] - L[i, :j] @ U[:j, j]
        if abs(L[j, j]) < tol:
            raise ValueError("Zero pivot: LU decomposition does not exist")
        for i in range(j + 1, n):                # row j of U
            U[j, i] = (A[j, i] - L[j, :j] @ U[:j, i]) / L[j, j]
    return L, U
 
 
def forward_substitution(L, b):
    """Solve L y = b where L is lower triangular."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y
 
 
def back_substitution(U, y):
    """Solve U x = y where U is upper triangular."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x
 
 
def solve_lu(A, b, method="doolittle"):
    b = np.array(b, dtype=float)
    L, U = lu_doolittle(A) if method == "doolittle" else lu_crout(A)
    y = forward_substitution(L, b)               # L y = b
    x = back_substitution(U, y)                  # U x = y
    return x, L, U, y
 
 
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
 
    # Example of the report: exact solution (1, -2, 3)
    A = [[1, 2, 1], [2, 3, 1], [5, 6, 9]]
    b = [0, -1, 20]
 
    for method in ("doolittle", "crout"):
        x, L, U, y = solve_lu(A, b, method)
        print(f"--- {method.capitalize()} ---")
        print("L =\n", L)
        print("U =\n", U)
        print("y =", y)
        print("x =", x)
        print("Check L @ U == A:", np.allclose(L @ U, A))
 
