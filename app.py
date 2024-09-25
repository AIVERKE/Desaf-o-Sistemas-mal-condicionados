import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Matriz de Hilbert 3x3
A = np.array([[1, 1/2, 1/3],
              [1/2, 1/3, 1/4],
              [1/3, 1/4, 1/5]])

# Vector de resultados (b)
b = np.array([1, 1, 1])

# Resolver el sistema Ax = b
x = np.linalg.solve(A, b)

# Mostrar la solución
print(f"Solución del sistema: x = {x}")

# Calcular el determinante de A
det_A = np.linalg.det(A)
print(f"Determinante de A: {det_A}")

# Calcular el número de condición de A
cond_A = np.linalg.cond(A)
print(f"Número de condición de A: {cond_A}")

# Calcular la matriz identidad aproximada (A * inv(A))
inv_A = np.linalg.inv(A)
identity_approx = np.dot(A, inv_A)
print("Matriz identidad aproximada:\\n", identity_approx)

# Graficar los planos correspondientes a las ecuaciones
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Definir un rango para los valores de x e y
x_vals = np.linspace(-1, 1, 100)
y_vals = np.linspace(-1, 1, 100)
x_vals, y_vals = np.meshgrid(x_vals, y_vals)

# Definir las ecuaciones de los planos (z = f(x, y)) para cada ecuación
z1 = (b[0] - A[0, 0] * x_vals - A[0, 1] * y_vals) / A[0, 2]
z2 = (b[1] - A[1, 0] * x_vals - A[1, 1] * y_vals) / A[1, 2]
z3 = (b[2] - A[2, 0] * x_vals - A[2, 1] * y_vals) / A[2, 2]

# Graficar los tres planos
ax.plot_surface(x_vals, y_vals, z1, alpha=0.5, rstride=100, cstride=100, color='red')
ax.plot_surface(x_vals, y_vals, z2, alpha=0.5, rstride=100, cstride=100, color='blue')
ax.plot_surface(x_vals, y_vals, z3, alpha=0.5, rstride=100, cstride=100, color='green')

# Etiquetas de los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title("Planos de las Ecuaciones del Sistema de Hilbert 3x3")
plt.show()
