import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

M = 40  # Capacidad del inventario
N = 100  # Número de nodos

lbd = 0.1  # Parámetro de la distribución exponencial
p = 3      # Costo unitario por demanda insatisfecha
h = 0.5    # Costo unitario por almacenamiento
c = 1.5    # Costo unitario de producción

alpha = 0.6       # Factor de descuento
epsilon = 0.001   # Criterio de convergencia
max_iter = 1000   # Número máximo de iteraciones

# Nodos
S = np.linspace(0, M, N)

# Política inicial
f = np.zeros(N)

# Función de valor anterior
V_anterior = None

# Vector indicador para el nodo cero
ell_vec = np.isclose(S, 0).astype(float)


# Función para calcular una fila de la matriz de transición
def calcular_P_a(z):

    P_a = np.zeros(N)
    exp_z = np.exp(-lbd * z)

    for j in range(N):

        if j == 0:

            A = max(0, z - S[j + 1])
            B = max(0, z - S[j])

            exp_A = np.exp(-lbd * A)
            exp_B = np.exp(-lbd * B)

            P_a[j] = (
                ell_vec[j] * exp_z
                +
                1 / (S[j + 1] - S[j])
                * (
                    (S[j + 1] - z)
                    * (exp_A - exp_B)
                    +
                    (A + 1 / lbd) * exp_A
                    -
                    (B + 1 / lbd) * exp_B
                )
            )

        elif j == N - 1:

            A = max(0, z - S[N - 1])
            B = max(0, z - S[N - 2])

            exp_A = np.exp(-lbd * A)
            exp_B = np.exp(-lbd * B)

            P_a[j] = (
                ell_vec[j] * exp_z
                +
                1 / (S[N - 1] - S[N - 2])
                * (
                    (z - S[N - 2])
                    * (exp_A - exp_B)
                    -
                    (A + 1 / lbd) * exp_A
                    +
                    (B + 1 / lbd) * exp_B
                )
            )

        else:

            A_1 = max(0, z - S[j + 1])
            B_1 = max(0, z - S[j])

            A_2 = B_1
            B_2 = max(0, z - S[j - 1])

            exp_A1 = np.exp(-lbd * A_1)
            exp_B1 = np.exp(-lbd * B_1)
            exp_A2 = np.exp(-lbd * A_2)
            exp_B2 = np.exp(-lbd * B_2)

            I_1 = ell_vec[j] * exp_z

            I_2 = (
                1 / (S[j + 1] - S[j])
                * (
                    (S[j + 1] - z)
                    * (exp_A1 - exp_B1)
                    +
                    (A_1 + 1 / lbd) * exp_A1
                    -
                    (B_1 + 1 / lbd) * exp_B1
                )
            )

            I_3 = (
                1 / (S[j] - S[j - 1])
                * (
                    (z - S[j - 1])
                    * (exp_A2 - exp_B2)
                    -
                    (A_2 + 1 / lbd) * exp_A2
                    +
                    (B_2 + 1 / lbd) * exp_B2
                )
            )

            P_a[j] = I_1 + I_2 + I_3

    return P_a


# Iteración de políticas

for iteration in range(max_iter):

    # Costo asociado a la política actual
    Cf = c * f + h * (S + f) + (p / lbd) * np.exp(-lbd * (S + f))

    # Matriz de transición P asociada a la política actual
    z_f = S + f
    P = np.zeros((N, N))

    for i in range(N):
        P[i, :] = calcular_P_a(z_f[i])

    # Resolvemos el sistema C = (I - alpha P)V
    V = np.linalg.solve(np.eye(N) - alpha * P, Cf)

    # Comparación de funciones de valor
    if V_anterior is None:

        diferencia_V = np.inf
        texto_diferencia = "no disponible en la primera iteración"

    else:

        diferencia_V = np.linalg.norm(V - V_anterior, ord=np.inf)
        texto_diferencia = f"{diferencia_V:.6e}"

    # Mejora de política
    f_nueva = np.zeros(N)

    for i in range(N):

        limite_superior = M - S[i]

        if np.isclose(limite_superior, 0):
            f_nueva[i] = 0.0
            continue

        def G(a, i=i):

            z_a = S[i] + a

            Ca_i = c * a + h * z_a + (p / lbd) * np.exp(-lbd * z_a)

            P_z = calcular_P_a(z_a)

            suma = np.dot(P_z, V)

            return Ca_i + alpha * suma

        resultado = minimize_scalar(
            G,
            bounds=(0, limite_superior),
            method="bounded",
            options={"xatol": 1e-10}
        )

        f_nueva[i] = resultado.x

    # Impresión de resultados de la iteración
    print("Iteración:", iteration + 1)

    print("Política obtenida en esta iteración:")
    print(f_nueva)

    print("Función de valor:")
    print(V)

    print("Comparación de funciones de valor en norma infinito:")
    print(texto_diferencia)

    print("-" * 50)

    # Criterio de convergencia
    if diferencia_V < epsilon:
        print("La función de valor convergió en norma infinito.")
        f = f_nueva.copy()
        break

    # Actualizamos para la siguiente iteración
    V_anterior = V.copy()
    f = f_nueva.copy()

else:
    print("Se alcanzó el número máximo de iteraciones.")


# Resultados finales
print("Última política:")
print(f)

print("Función de valor final:")
print(V)