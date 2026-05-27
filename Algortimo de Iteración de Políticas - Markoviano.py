# Algortimo de Iteración de Políticas - Markoviano

import numpy as np
import matplotlib.pyplot as plt

M = 2 # Capacidad del inventario
N = 3 # Número de nodos
S = [0]
s = 0
# Nodos

for i in range(N-1):
    s = s + M/(N-1)
    S.append(s)
    print(S)

# Elegimos una política inicial

f_0 = []
for i in range(N):
    f_0.append(0)
print("Política inicial: ", f_0)

# Calculamos el vector de costos

def I(j):
    if j >= 2:


def ell(j,x):
    
