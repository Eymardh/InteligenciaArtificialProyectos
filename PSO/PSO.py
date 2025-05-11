import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Función objetivo: Sphere (mínimo en (0,0))
def sphere(x):
    return np.sum(x**2)

# Clase Partícula
class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position.copy()
        self.best_fitness = sphere(self.position)
    
    def update_velocity(self, global_best, w, c1, c2):
        r1, r2 = np.random.rand(2)
        self.velocity = w * self.velocity + \
                       c1 * r1 * (self.best_position - self.position) + \
                       c2 * r2 * (global_best - self.position)
    
    def update_position(self, bounds):
        self.position += self.velocity
        # Asegurar que se mantenga dentro de los límites
        self.position = np.clip(self.position, bounds[0], bounds[1])

# Parámetros PSO
n_particles = 30
n_iterations = 100
dim = 2  # Dimensión del problema
bounds = (-5, 5)  # Límites del espacio de búsqueda
w = 0.9  # Inercia inicial
c1, c2 = 2, 2  # Coeficientes cognitivo y social

# Inicializar enjambre
swarm = [Particle(dim, bounds) for _ in range(n_particles)]
global_best = min(swarm, key=lambda p: p.best_fitness).best_position.copy()
global_best_fitness = sphere(global_best)

# Historial para visualización
history = []

# Ejecutar PSO
for i in range(n_iterations):
    for particle in swarm:
        fitness = sphere(particle.position)
        
        # Actualizar mejor local
        if fitness < particle.best_fitness:
            particle.best_position = particle.position.copy()
            particle.best_fitness = fitness
        
        # Actualizar mejor global
        if fitness < global_best_fitness:
            global_best = particle.position.copy()
            global_best_fitness = fitness
        
        # Actualizar velocidad y posición
        particle.update_velocity(global_best, w, c1, c2)
        particle.update_position(bounds)
    
    # Reducir inercia linealmente
    w = 0.9 - (0.5 * i / n_iterations)
    
    history.append(global_best_fitness)
    print(f"Iteración {i+1}: Mejor fitness = {global_best_fitness:.4f}")

# Visualización
plt.figure(figsize=(10, 4))
plt.plot(history, 'r-', linewidth=2)
plt.title("Convergencia del PSO")
plt.xlabel("Iteración")
plt.ylabel("Mejor Fitness (Sphere)")
plt.grid(True)

# Gráfico 3D de la función y partículas
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(bounds[0], bounds[1], 50)
y = np.linspace(bounds[0], bounds[1], 50)
X, Y = np.meshgrid(x, y)
Z = sphere([X, Y])
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Posiciones finales de las partículas
particles_pos = np.array([p.position for p in swarm])
ax.scatter(particles_pos[:,0], particles_pos[:,1], 
           [sphere(p) for p in particles_pos], 
           c='red', s=50, label='Partículas')
ax.set_title("Espacio de Búsqueda y Partículas")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Sphere(X,Y)")
plt.legend()
plt.show()