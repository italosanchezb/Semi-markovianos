import numpy as np
from scipy.optimize import minimize_scalar

# Parámetros del modelo
M = 40      # Capacidad del inventario
lbd = 0.1   # Parámetro de la distribución exponencial
p = 3       # Costo unitario por demanda insatisfecha
h = 0.5     # Costo unitario por almacenamiento
c = 1.5     # Costo unitario de producción

epsilon = 0.001
max_iter = 1000


def construir_P(z_i, S, lbd):
    """
    Construye la matriz de transición para uno o varios valores z_i.
    Cada fila corresponde a un valor de z_i.
    """

    z_i = np.atleast_1d(z_i)
    K = len(z_i)
    N = len(S)

    ell_vec = np.isclose(S, 0).astype(float)

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


def calcular_fila_P(z, S, lbd):
    """
    Calcula una sola fila de transición.
    """
    return construir_P([z], S, lbd)[0, :]


def ejecutar_API(N, alpha, imprimir_iteraciones=False):
    """
    Ejecuta el algoritmo API para una malla con N nodos y factor alpha.
    """

    S = np.linspace(0, M, N)

    # Política inicial g_0(x) = 0
    f = np.zeros(N)

    V_anterior = None

    for k in range(max_iter + 1):

        # Costo de la política actual
        z_f = S + f
        Cf = c * f + h * z_f + (p / lbd) * np.exp(-lbd * z_f)

        # Matriz de transición de la política actual
        P = construir_P(z_f, S, lbd)

        # Evaluación de política: u_k
        V = np.linalg.solve(np.eye(N) - alpha * P, Cf)

        # Comparación ||u_k - u_{k-1}|| infinito
        if V_anterior is None:
            diferencia_V = np.inf
        else:
            diferencia_V = np.linalg.norm(V - V_anterior, ord=np.inf)

        # Punto de reorden de la política actual
        # Para una política base-stock, f(0) es el nivel objetivo.
        S_reorden = f[0]

        if imprimir_iteraciones:
            print(f"Iteración k = {k}")
            print(f"||u_{k} - u_{k-1}||_inf = {diferencia_V:.6e}" if k > 0 else "Comparación no disponible")
            print(f"Punto de reorden aproximado = {S_reorden:.6f}")
            print("-" * 50)

        # Criterio de paro
        if V_anterior is not None and diferencia_V < epsilon:
            return {
                "N": N,
                "alpha": alpha,
                "k": k,
                "diferencia_V": diferencia_V,
                "S_reorden": S_reorden,
                "V": V,
                "f": f,
                "S": S
            }

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

                P_z = calcular_fila_P(z_a, S, lbd)

                return Ca_i + alpha * np.dot(P_z, V)

            resultado = minimize_scalar(
                G,
                bounds=(0, limite_superior),
                method="bounded",
                options={"xatol": 1e-10}
            )

            f_nueva[i] = resultado.x

        # Actualizamos para la siguiente iteración
        V_anterior = V.copy()
        f = f_nueva.copy()

    return {
        "N": N,
        "alpha": alpha,
        "k": max_iter,
        "diferencia_V": diferencia_V,
        "S_reorden": S_reorden,
        "V": V,
        "f": f,
        "S": S
    }


def imprimir_tabla_articulo(resultados, alpha):
    """
    Imprime una tabla parecida a la Tabla 1 del artículo.
    """

    Ns = [r["N"] for r in resultados]
    ks = [r["k"] for r in resultados]
    difs = [r["diferencia_V"] for r in resultados]
    puntos = [r["S_reorden"] for r in resultados]

    print()
    print(f"Tabla estilo artículo para alpha = {alpha}")
    print("-" * 80)

    print("N".ljust(28), end="")
    for N in Ns:
        print(f"{N:>14}", end="")
    print()

    print("k de paro".ljust(28), end="")
    for k in ks:
        print(f"{k:>14}", end="")
    print()

    print("||u_k^N - u_{k-1}^N||".ljust(28), end="")
    for d in difs:
        print(f"{d:>14.3e}", end="")
    print()

    print("S_hat_{alpha,k}^N".ljust(28), end="")
    for s in puntos:
        print(f"{s:>14.6f}", end="")
    print()

    print("-" * 80)


# Valores como en la Tabla 1 del artículo
N_values = [100, 500, 1000, 2000]
alpha_values = [0.6, 0.99]

for alpha in alpha_values:

    resultados_alpha = []

    for N in N_values:

        print(f"Ejecutando API con alpha = {alpha}, N = {N}...")

        resultado = ejecutar_API(
            N=N,
            alpha=alpha,
            imprimir_iteraciones=False
        )

        resultados_alpha.append(resultado)

    imprimir_tabla_articulo(resultados_alpha, alpha)