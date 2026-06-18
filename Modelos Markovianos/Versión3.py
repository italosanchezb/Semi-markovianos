# Algoritmo de Iteración de Políticas - Markoviano

import numpy as np
from scipy.optimize import minimize_scalar

M = 40       # Capacidad del inventario
N = 2000        # Número de nodos

lbd = 0.1    # Parámetro de la distribución
p = 3        # Costo unitario por demanda insatisfecha
h = 0.5      # Costo unitario por almacenamiento
c = 1.5      # Costo unitario de producción

alpha = 0.99      # Factor de descuento
epsilon = 0.001   # Criterio de convergencia
max_iter = 1000   # Número máximo de iteraciones

# Nodos
S = np.linspace(0, M, N)

# Política inicial
f = np.zeros(N)

# Vector ell
ell_vec = np.isclose(S, 0).astype(float)


# Función para calcular varias filas de transición al mismo tiempo
def P_f(z_i):

    z_i = np.atleast_1d(z_i)
    K = len(z_i)

    P = np.zeros((K, N))

    # Caso j = 0
    A = np.maximum(0, z_i - S[1])
    B = np.maximum(0, z_i - S[0])

    P[:, 0] = (
        ell_vec[0] * np.exp(-lbd * z_i)
        +
        1 / (S[1] - S[0])
        * (
            (S[1] - z_i)
            * (np.exp(-lbd * A) - np.exp(-lbd * B))
            +
            (A + 1 / lbd) * np.exp(-lbd * A)
            -
            (B + 1 / lbd) * np.exp(-lbd * B)
        )
    )

    # Caso j = N - 1
    A = np.maximum(0, z_i - S[N - 1])
    B = np.maximum(0, z_i - S[N - 2])

    P[:, N - 1] = (
        ell_vec[N - 1] * np.exp(-lbd * z_i)
        +
        1 / (S[N - 1] - S[N - 2])
        * (
            (z_i - S[N - 2])
            * (np.exp(-lbd * A) - np.exp(-lbd * B))
            -
            (A + 1 / lbd) * np.exp(-lbd * A)
            +
            (B + 1 / lbd) * np.exp(-lbd * B)
        )
    )

    # Casos interiores
    z = z_i[:, None]

    S_j_menos = S[:-2]
    S_j = S[1:-1]
    S_j_mas = S[2:]

    A_1 = np.maximum(0, z - S_j_mas)
    B_1 = np.maximum(0, z - S_j)

    A_2 = B_1
    B_2 = np.maximum(0, z - S_j_menos)

    I_1 = ell_vec[1:-1] * np.exp(-lbd * z)

    I_2 = (
        1 / (S_j_mas - S_j)
        * (
            (S_j_mas - z)
            * (np.exp(-lbd * A_1) - np.exp(-lbd * B_1))
            +
            (A_1 + 1 / lbd) * np.exp(-lbd * A_1)
            -
            (B_1 + 1 / lbd) * np.exp(-lbd * B_1)
        )
    )

    I_3 = (
        1 / (S_j - S_j_menos)
        * (
            (z - S_j_menos)
            * (np.exp(-lbd * A_2) - np.exp(-lbd * B_2))
            -
            (A_2 + 1 / lbd) * np.exp(-lbd * A_2)
            +
            (B_2 + 1 / lbd) * np.exp(-lbd * B_2)
        )
    )

    P[:, 1:-1] = I_1 + I_2 + I_3

    return P


# Función para calcular una sola fila de transición
def P_a(z):

    return P_f([z])[0, :]


# Iteración de políticas
convergio = False

V_anterior = None
f_anterior = None

for iteration in range(max_iter):

    # En esta iteración se evalúa la política actual f_k

    # Costo asociado a la política actual
    Cf = c * f + h * (S + f) + (p / lbd) * np.exp(-lbd * (S + f))

    # Matriz de transición asociada a la política actual
    P = P_f(S + f)

    # Función de valor asociada a la política actual
    V = np.linalg.solve(np.eye(N) - alpha * P, Cf)

    print("Iteración:", iteration + 1)

    # Comparación de funciones de valor: ||V_k - V_{k-1}|| infinito
    if V_anterior is None:

        print("Comparación de funciones de valor en norma infinito: no disponible en la primera iteración")

    else:

        diferencia_V = np.linalg.norm(V - V_anterior, ord=np.inf)

        print(f"Comparación de funciones de valor en norma infinito: {diferencia_V:.6e}")

    # Comparación de políticas: ||f_k - f_{k-1}|| infinito
    if f_anterior is None:

        print("Comparación de políticas en norma infinito: no disponible en la primera iteración")

    else:

        diferencia_f = np.linalg.norm(f - f_anterior, ord=np.inf)

        print(f"Comparación de políticas en norma infinito: {diferencia_f:.6e}")

        # Criterio de convergencia de la política
        if diferencia_f < epsilon:

            convergio = True

            print()
            print("La política convergió.")
            break

    print()

    # Guardamos la política y la función de valor actuales
    # antes de calcular la siguiente política
    f_anterior = f.copy()
    V_anterior = V.copy()

    # Mejora de política: calculamos f_{k+1}
    f_nueva = np.zeros(N)

    for i in range(N):

        limite_superior = M - S[i]

        if np.isclose(limite_superior, 0):

            f_nueva[i] = 0.0

            continue

        def G(a):

            z_a = S[i] + a

            Ca_i = c * a + h * z_a + (p / lbd) * np.exp(-lbd * z_a)

            # Corrección: no usamos el mismo nombre P_a para la variable
            P_a_vec = P_a(z_a)

            suma = np.dot(P_a_vec, V)

            return Ca_i + alpha * suma

        resultado = minimize_scalar(
            G,
            bounds=(0, limite_superior),
            method="bounded",
            options={"xatol": 1e-10}
        )

        if not resultado.success:

            raise RuntimeError(f"La optimización falló en el nodo i = {i}")

        f_nueva[i] = resultado.x

    # La política nueva será evaluada en la siguiente iteración
    f = f_nueva.copy()


if not convergio:

    print("Se alcanzó el número máximo de iteraciones sin convergencia.")


# Resultados finales
#print()
#print("Última política:")
#print(f)

#print()
#print("Última función de valor:")
#print(V)

# Nivel objetivo después de ordenar
nivel_objetivo = S + f

#print()
#print("Nivel objetivo S + f:")
#print(nivel_objetivo)

# Punto de reorden aproximado
# Buscamos los estados donde la política ordena una cantidad positiva
indices_orden = np.where(f > epsilon)[0]

if len(indices_orden) > 0:

    punto_reorden_malla = np.max(S[indices_orden])
    nivel_base = np.median((S + f)[indices_orden])

    print("Punto de reorden aproximado en la malla:")
    print(punto_reorden_malla)

    print()
    print("Nivel base aproximado:")
    print(nivel_base)

else:

    print("No hay punto de reorden: la política nunca ordena.")

# Nivel máximo al que se lleva el inventario después de ordenar
nivel_maximo_despues_de_ordenar = np.max(nivel_objetivo)

print()
print("Nivel máximo después de ordenar:")
print(nivel_maximo_despues_de_ordenar)