# Algortimo de Iteración de Políticas - Markoviano

import numpy as np
import matplotlib.pyplot as plt

M = 2 # Capacidad del inventario
N = 3 # Número de nodos

lbd = 0.1 # Parámetro de la distribución de Poisson
p = 3 # Costo unitario por demanda insatisfecha
h = 0.5 # Costo unitario por almacenamiento
c = 1.5 # Costo unitario de producción

alpha = 0.1 # Factor de descuento
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

    P = np.zeros((N, N))
    P = np.ones((N, N))

    # Resolvemos el sistema C= (I-alpha P)V

    # Definimos las variables
    I = np.eye(N)
    U = np.zeros(N)
    epsilon  =  0.001
    alpha = 0.1

    # Iteramos

    while True:
        U_m = Cf + alpha * np.dot(P, U)
        if np.linalg.norm(U_m - U,ord=np.inf) < epsilon:
            U = U_m
            print("Valor de U: ", U)
            break
        U = U_m
        print("Valor de U: ", U)

    V = U
    print("Valor de V: ", V)

    # Actualizamos la política

    f_1 = []
    for i in range(N):
        A_i = [0] 
        a = 0
        G_i = []
        for k in range(N-1):
            a = a + (M-S[i])/(N-1)
            A_i.append(a)
        for k in range(N):
            Ca_ik = c * A_i[k] + h * (S[i] + A_i[k]) + p/lbd * float(np.exp(-lbd*(S[i] + A_i[k])))
            suma = 0
            for j in range(N):
                suma = suma + P[i][j]*V[j]
            G_i.append(float(Ca_ik + alpha * suma))
        f_1.append(A_i[np.argmin(G_i)])
        print(A_i, G_i,f_1)

    f_0 = f_1
