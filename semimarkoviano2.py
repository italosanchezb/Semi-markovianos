import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

M = 10.358  # Cota del conjunto
N = 100 # Número de nodos

alpha = 0.2 # Factor de descuento
epsilon = 1e-18 # Criterio de convergencia
max_iter = 1000 # Número máximo de iteraciones  
b = 0.05 # Parámetro de la función de supervivencia
q = 0.5

# Distrubución Gamma
beta = 5    # Parámetro de forma
gma = 1/2     # Parámetro de escala

def gamma_cdf(x):
    return sp.stats.gamma.cdf(x, a=beta, scale=gma)

# Distribución Weibull
rho = 4 # Parámetro de forma
lbd = 1/5   # Parámetro de escala

def weibull_densidad(x):
    return (rho/lbd)*(x/lbd)**(rho-1)*np.exp(-(x/lbd)**rho)

# Costos
K_1 = 15 # Costo de reemplazo por falla
K_2 = 10 # Costo de reemplazo programado
K_3 = 1 # Costo proporcional al daño


# Nodos
S = np.linspace(0, M, N)

# Política inicial
f = np.full(N,1)

# Variables para comparar
f_anterior = None
V_anterior = None

# Criterio de convergencia
converge = False

# Límites de las acciones admisibles
theta_1 = sp.stats.gamma.ppf(0.05,a=beta,scale=gma)
theta_2 = sp.stats.gamma.ppf(0.95,a=beta,scale=gma)

print(theta_1,theta_2)



def h(u):
    return np.exp(-b*lbd*u**(1/rho)-u)

val,error = sp.integrate.quad(h,0,np.inf)
int_1 = val

for iteracion in range(max_iter):

    # Función de costo

    def C(x,a):
        return K_1 * gamma_cdf(a) *(1-(np.exp(-b*x))*int_1) + K_2 * (1-gamma_cdf(a)) + K_3 * x

    C_tilde = np.zeros(N)

    for i in range(N):

        a = f[i]
        C_tilde[i] = C(S[i],a)

    # print("Vector de costos: ", C_tilde)
    # Kernel de transición

    P = np.zeros((N, N))

    ## Funciones auxiliares

    def A_i(a):
        kappa = alpha + 1/gma
        return (1/(gma*kappa)**beta)*sp.special.gammainc(beta,kappa*a)


    def D_i(x):
        return 1 - np.exp(-b*x)*int_1


    def E_i(a):
        return np.exp(-alpha * a)*(sp.special.gammaincc(beta,a/gma))


    def B1(si,sj,sj1):
        def integrando(z):
            return (si + z - sj1)/(sj-sj1) * np.exp(-b*(si+z)) * weibull_densidad(z)
        liminf = max(0,sj1-si)
        limsup = max(0,sj-si)
        val_B_1, error_B_1 = sp.integrate.quad(integrando,liminf,limsup)
        return val_B_1


    def B2(si,sj,sj1):
        def integrando(z):
            return (sj1 - si - z)/(sj1 - sj) * np.exp(-b*(si+z)) * weibull_densidad(z)
        liminf = max(0,sj-si)
        limsup = max(0,sj1-si)
        val_B_1, error_B_1 = sp.integrate.quad(integrando,liminf,limsup)
        return val_B_1


    def K_i(a):
        def integrando(z):  
            return (S[1] - a - z)/(S[1]-S[0]) * np.exp(-b*(a + z)) * weibull_densidad(z)
        limsup = max(0,S[1] - a)
        val_K, error_K = sp.integrate.quad(integrando,0,limsup)
        return val_K


    def P_fila(i,a):
        fila = np.zeros(N)

        Ai = A_i(a)
        Ei = E_i(a)
        Di = D_i(S[i])
        Ki = K_i(S[i])
        # Caso j=0
        fila[0]  = Ai*Ki + Ai*Di + Ei

        # Casos j=1,...,N-2
        for j in range(1,N-1):
            B_1 = B1(S[i], S[j], S[j-1])
            B_2 = B2(S[i],S[j],S[j+1])
            fila[j] = Ai*(B_1+B_2)
        
        # Caso j=N-1
        B_3 = B1(S[i],S[N-1],S[N-2])
        fila[N-1] = Ai*B_3

        return fila


    def matriz_P(f):
        P = np.zeros((N,N))

        for i in range(N):
            P[i,:] = P_fila(i,f[i])
        
        return P

    P = matriz_P(f)

    # Resolvemos el sistema  lineal y estimamos las últimas 2 estimaciones de V

    V = np.linalg.solve(np.eye(N) - P, C_tilde)

    print("Iteración:", iteracion + 1)

    if V_anterior is not None:

        W = np.exp(q*S)

        argumento = np.abs(V_anterior-V)/W

        norma = np.max(argumento)

        x_sup = S[np.argmax(argumento)]

        print(f'||V_{iteracion+1} - V_{iteracion}|| = {norma}')
    
    else:
        print("Primera iteración")

    # Actualizamos la política

    f_nueva = np.zeros(N)

    for i in range(N):
        def G_i(a):

            return C(S[i],a) + np.dot(P_fila(i,a),V)

        min = sp.optimize.minimize_scalar(G_i, bounds=(theta_1, theta_2), method='bounded', options={'xatol': 1e-10})

        if not min.success:
            raise RuntimeError(f"Optimización fallida en el nodo {i} en la iteración {iteracion + 1}")
        
        f_nueva[i] = min.x
    

    

    diference_f = np.linalg.norm(f_nueva-f, ord=np.inf)

    print(f'||f_{iteracion+1} - f_{iteracion}|| = {diference_f}')

    if diference_f < epsilon and norma < epsilon:

        converge = True
        f_anterior = f.copy()
        V_anterior = V.copy()
        f = f_nueva.copy()

        print()
        print("Convergencia alcanzada en la iteración", iteracion + 1)
        
        break
    
    print()

    f_anterior = f.copy()
    V_anterior = V.copy()

    # Se actualiza la política para la siguiente iteración
    f = f_nueva.copy()

if not converge:
    print("No se alcanzó convergencia después de", max_iter, "iteraciones.")