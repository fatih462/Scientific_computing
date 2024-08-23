import numpy as np

def forward_subs(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = len(b)
    x = np.zeros(m)

    if L.shape[0] != L.shape[1] or L.shape[0] != len(b):
        raise ValueError("geht nicht")
    
    rang = 0
    for i in range(len(b)):
        if np.any(L[i, :] != 0):
            rang += 1
    if rang != L.shape[0]:
        raise ValueError("noe")

    for i in range(m):
        summe = 0
        if L[i, i] == 0:
            raise ValueError("noe")
        for k in range(1, i):
            summe += L[i, k] * x[k]
        x[i] = 1 / L[i, i] * (b[i] - summe)
    
    return x

def test_forward_subs():
    # Testmatrix
    L = np.array([[2, 0, 0],
                  [3, 4, 0],
                  [5, 6, 7]])

    # Testvektor
    b = np.array([1, 2, 3])

    try:
        x = forward_subs(L, b)
        print("Ergebnis der Vorwärtssubstitution:", x)
    except ValueError as e:
        print("Fehler:", e)
    print(np.linalg.solve(L, b))
    print(forward_subs(L, b))

test_forward_subs()