import numpy as np

# Funktion zur Berechnung der Amplituden
def calculate_coefficients(x_values, y_values, N):
    coefficients = []
    coefficients.append(np.mean(y_values))
    for n in range(1, N+1):
        coef = 2 * np.mean([y * np.cos(2 * np.pi * n * x) for x, y in zip(x_values, y_values)])
        coefficients.append(coef)
    return coefficients

# Gegebene Werte
x_values = np.arange(1, 16)
y_values = np.array([0.25] * 15)  # Wert ist überall 0.25

# Anzahl der Perioden
N = 15

# Berechnung der Koeffizienten
coefficients = calculate_coefficients(x_values, y_values, N)

# Ausgabe der Koeffizienten
print("Koeffizienten:")
for i, coef in enumerate(coefficients):
    print("a_{} = {:.2f}".format(i, coef))

