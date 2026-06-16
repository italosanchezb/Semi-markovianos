import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

M = 40  # Cota del conjunto
N = 10 # Número de nodos

alpha = 0.99 # Factor de descuento
epsilon = 0.001 # Criterio de convergencia
max_iter = 1000 # Número máximo de iteraciones  
b = 0.4 # Parámetro de la función de supervivencia

# Distrubución Gamma
beta = 2    # Parámetro de forma
gma = 3     # Parámetro de escala

def gamma_cdf(x):
    return sp.stats.gamma.cdf(x, a=beta, scale=gma)

# Distribución Weibull
rho = 1.5 # Parámetro de forma
lbd = 2   # Parámetro de escala

def weibull_densidad(x):
    return (rho/lbd)*(x/lbd)**(rho-1)*np.exp(-(x/lbd)**rho)

# Costos
K_1 = 1.5 # Costo de reemplazo por falla
K_2 = 0.5 # Costo de reemplazo programado
K_3 = 0.1 # Costo proporcional al daño


# Nodos
S = np.linspace(0, M, N)

# Política inicial
f = np.full(N,0.5)

# Variables para comparar
f_anterior = None
V_anterior = None

# Criterio de convergencia
converge = False

# Límites de las acciones admisibles
theta_1 = 0.5
theta_2 = 1




def h(u):
    return np.exp(-b*lbd*u**(1/rho)-u)

val,error = sp.integrate.quad(h,0,np.inf)
int_1 = val

for iteracion in range(max_iter):

    def C(x,a):
        return K_1 * gamma_cdf(a) *(1-(np.exp(-b*x))*int_1) + K_2 * (1-gamma_cdf(a)) + K_3 * x

    C_tilde = np.zeros(N)

    # Kernel

    P = np.zeros((N, N))

    ## Funciones auxiliares

    def A_i(a):
        kappa = alpha + 1/gma
        return (1/(gma*kappa)**beta)*sp.special.gammainc(beta,kappa*a)

    def D_i(x):
        return 1 - np.exp(-b*x)*int_1

    def E_i(a):
        return np.exp(-alpha * a)*(sp.special.gammaincc(beta,a/gma))

    def B1_i(si,sj,sj1):
        def integrando(z):
            return (si + z - sj1)/(sj-sj1) * np.exp(-b*(si+z)) * weibull_densidad(z)
        liminf = max(0,sj1-si)
        limsup = max(0,sj-si)
        val_B_1, error_B_1 = sp.integrate.quad(integrando,liminf,limsup)
        return val_B_1


    def B2_i(si,sj,sj1):
        def integrando(z):
            return (sj1 - si - z)/(sj1 - sj) * np.exp(-b*(si+z)) * weibull_densidad(z)
        liminf = max(0,sj-si)
        limsup = max(0,sj1-si)
        val_B_1, error_B_1 = sp.integrate.quad(integrando,liminf,limsup)
        return val_B_1

    A = np.zeros(N)
    B_1 = np.zeros((N,N))
    B_2 = np.zeros((N,N))
    D = np.zeros(N)
    E = np.zeros(N)

    for i in range(N):
        A[i] = A_i(f[i])
        D[i] = D_i(S[i])
        E[i] = E_i(f[i])
        for j in range(1,N-1):
            B_1[i,j] = B1_i(S[i],S[j],S[j-1])
            B_2[i,j] = B2_i(S[i],S[j],S[j+1])
            P[i,j] = A[i] * (B_1[i,j] + B_2[i,j])


    # Caso j=0 y j=N-1

    def K_i(a):
        def integrando(z):  
            return (S[1] - a - z)/(S[1]-S[0]) * np.exp(-b*(a + z)) * weibull_densidad(z)
        limsup = max(0,S[1] - a)
        val_K, error_K = sp.integrate.quad(integrando,0,limsup)
        return val_K

    K = np.zeros(N)

    for i in range(N):
        K[i] = K_i(S[i])
        P[i,0] = A[i]*K[i] + A[i]*D[i] + E[i]
        B_1[i,N-1] = B1_i(S[i],S[N-1],S[N-2])
        P[i,N-1] = A[i] * B_1[i,N-1]


    for i in range(N):

        a = f[i]
        C_tilde[i] = C(S[i],a)

    # print("Vector de costos: ", C_tilde)

    V = np.linalg.solve(np.eye(N) - P, C_tilde)

    print("Iteración:", iteracion + 1)

    if V_anterior is not None:

        diference_V =  np.linalg.norm(V-V_anterior, ord=np.inf)

        print(f'||V_{iteracion+1} - V_{iteracion}|| = {diference_V}')
    
    else:
        print("Primera iteración")

    if f_anterior is not None:

        diference_f = np.linalg.norm(f-f_anterior, ord=np.inf)

        print(f'||f_{iteracion+1} - f_{iteracion}|| = {diference_f}')

        if diference_f < epsilon:

            converge = True

            print()
            print("Convergencia alcanzada en la iteración", iteracion + 1)
            break

    else:
        print("Primera iteración")
    
    print()

    f_anterior = f.copy()
    V_anterior = V.copy()

    f_nueva = np.zeros(N)

    for i in range(N):
        def G_i(a):

            return C(S[i],a) + np.dot(P[i,:],V)

        min = sp.optimize.minimize_scalar(G_i, bounds=(theta_1, theta_2), method='bounded', options={'xatol': 1e-10})

        if not min.success:
            raise RuntimeError(f"Optimización fallida en el nodo {i} en la iteración {iteracion + 1}")
        
        f_nueva[i] = min.x
    
    # Se actualiza la política para la siguiente iteración
    f = f_nueva.copy()

if not converge:
    print("No se alcanzó convergencia después de", max_iter, "iteraciones.")