# Algortimo de Iteración de Políticas - Markoviano

import numpy as np
import matplotlib.pyplot as plt

M = 2 # Capacidad del inventario
N = 3 # Número de nodos

lbd = 0.1 # Parámetro de la distribución de Poisson
p = 3 # Costo unitario por demanda insatisfecha
h = 0.5 # Costo unitario por almacenamiento
c = 1.5 # Costo unitario de producción

alpha = 0.6 # Factor de descuento
epsilon = 0.001 # Criterio de convergencia
max_iter = 1000 # Número máximo de iteraciones

# Nodos
S = np.linspace(0, M, N)

# Política inicial
f = np.zeros(N)

# Calculamos el vector de costos C

for iteration in range(max_iter):
    
    Cf = c * f + h * (S + f) + (p / lbd) * np.exp(-lbd * (S + f))

    print("Vector de costos: ", Cf)

    # Matriz de transición P
    def ell(S, j):
        if np.isclose(S[j], 0):
            return 1.0
        return 0.0

    def calcular_P_a(z):

        P_a = np.zeros(N)

        for j in range(N):

            if j == 0:
                # Caso j = 1

                A = np.max([0, z - S[j + 1]])
                B = np.max([0, z - S[j]])

                P_a[j] = (
                    ell(S, j) * np.exp(-lbd * z)
                    +
                    1 / (S[j + 1] - S[j])
                    * (
                        (S[j + 1] - z)
                        * (np.exp(-lbd * A) - np.exp(-lbd * B))
                        +
                        (A + 1 / lbd) * np.exp(-lbd * A)
                        -
                        (B + 1 / lbd) * np.exp(-lbd * B)
                    )
                )

            elif j == N - 1:
                # Caso j = N

                A = np.max([0, z - S[N - 1]])
                B = np.max([0, z - S[N - 2]])

                P_a[j] = (
                    ell(S, j) * np.exp(-lbd * z)
                    +
                    1 / (S[N - 1] - S[N - 2])
                    * (
                        (z - S[N - 2])
                        * (np.exp(-lbd * A) - np.exp(-lbd * B))
                        -
                        (A + 1 / lbd) * np.exp(-lbd * A)
                        +
                        (B + 1 / lbd) * np.exp(-lbd * B)
                    )
                )

            else:
                # Casos interiores

                A_1 = np.max([0, z - S[j + 1]])
                B_1 = np.max([0, z - S[j]])

                A_2 = B_1
                B_2 = np.max([0, z - S[j - 1]])

                I_1 = ell(S, j) * np.exp(-lbd * z)

                I_2 = (
                    1 / (S[j + 1] - S[j])
                    * (
                        (S[j + 1] - z)
                        * (np.exp(-lbd * A_1) - np.exp(-lbd * B_1))
                        +
                        (A_1 + 1 / lbd) * np.exp(-lbd * A_1)
                        -
                        (B_1 + 1 / lbd) * np.exp(-lbd * B_1)
                    )
                )

                I_3 = (
                    1 / (S[j] - S[j - 1])
                    * (
                        (z - S[j - 1])
                        * (np.exp(-lbd * A_2) - np.exp(-lbd * B_2))
                        -
                        (A_2 + 1 / lbd) * np.exp(-lbd * A_2)
                        +
                        (B_2 + 1 / lbd) * np.exp(-lbd * B_2)
                    )
                )

                P_a[j] = I_1 + I_2 + I_3

        return P_a

    P = np.zeros((N, N))

    for i in range(N):

        z_i = S[i] + f[i]

        P[i, :] = calcular_P_a(z_i)

    print("Matriz de transición P: ", P)

    # Resolvemos el sistema C= (I-alpha P)V

    V = np.linalg.solve(np.eye(N) - alpha * P, Cf)

    f_nueva = np.zeros(N)

    for i in range(N):

        A_i = np.linspace(0, M - S[i], N)

        G_i = []

        for a in A_i:

            z_a = S[i] + a

            Ca_i = c * a + h * z_a + (p / lbd) * np.exp(-lbd * z_a)

            P_a = calcular_P_a(z_a)

            suma = np.dot(P_a, V)

            G_i.append(Ca_i + alpha * suma)

        f_nueva[i] = A_i[np.argmin(G_i)]

    print("Iteración:", iteration + 1)
    print("Política:", f_nueva)
    print("Valor V:", V)
    print()

    if np.max(np.abs(f_nueva - f)) < epsilon:
        print("La política convergió.")
        break

    f = f_nueva