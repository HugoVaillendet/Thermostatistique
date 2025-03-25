import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from itertools import product

class Gas:
    def __init__(self, N, mass, radius, L, v_0, duration, steps):
        self.N = N
        self.mass = mass
        self.radius = radius
        self.L = L
        self.v_0 = v_0
        self.duration = duration
        self.steps = steps
        self.dt = duration/steps

        grid=int(np.ceil(np.sqrt(N)))
        space=L/grid
        x=np.linspace(radius+space/2, L-radius - space/2, grid)
        pos=list(product(x, x))

        self.r=np.array(pos[:N])

        theta=np.random.uniform(0, 2*np.pi, size=N)
        v_x,v_y=self.v_0*np.cos(theta), self.v_0*np.sin(theta)
        self.velocity=np.stack((v_x,v_y), axis=1)

    def collisions(self):
        r_next = self.r + self.velocity * self.dt

        self.velocity[r_next[:, 0] < self.radius, 0] *= -1
        self.velocity[r_next[:, 0] > self.L - self.radius, 0] *= -1
        self.velocity[r_next[:, 1] < self.radius, 1] *= -1
        self.velocity[r_next[:, 1] > self.L - self.radius, 1] *= -1

        for i in range(self.N):
            for j in range(i + 1, self.N):
                if np.linalg.norm(r_next[i] - r_next[j]) < 2 * self.radius:
                    delta_velocity = self.velocity[i] - self.velocity[j]
                    delta_r = self.r[i] - self.r[j]

                    self.velocity[i] -= delta_r.dot(delta_velocity) / (delta_r.dot(delta_r)) * delta_r
                    self.velocity[j] += delta_r.dot(delta_velocity) / (delta_r.dot(delta_r)) * delta_r

    def step(self):
        self.collisions()
        self.r += self.velocity * self.dt

    def simulation(self):
        positions = np.zeros((self.steps, self.N, 2))
        velocities = np.zeros((self.steps, self.N))

        for n in range(self.steps):
            self.step()
            positions[n, :, :] = self.r
            velocities[n, :] = np.linalg.norm(self.velocity, axis=1)

        return positions, velocities

N=100
mass=1
radius=0.2
L=20
v_0=3
duration=10
steps=750

gas=Gas(N, mass, radius, L, v_0, duration, steps)

"""fig=plt.figure()
ax=fig.add_subplot(111)
for(x,y) in gas.r:
    ax.add_artist(plt.Circle((x,y), r, color='r'))
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')

plt.show()"""

positions, velocities = gas.simulation()

fig,ax1= plt.subplots(1, figsize = (12,6))

def update(frame):
    ax1.clear()
    for i in range(gas.N):
        x, y = positions[frame,i, 0], positions[frame, i, 1]
        circle = plt.Circle((x, y), gas.radius, fill=True)
        ax1.add_artist(circle)

    ax1.set_xlabel('$x$', fontsize=15)
    ax1.set_ylabel('$y$', fontsize=15)
    ax1.set_title('Ideal gas animation', fontsize=15)
    ax1.set_xlim(0, gas.L)
    ax1.set_ylim(0, gas.L)
    ax1.set_aspect('equal')
    ax1.set_xticks([]) #remove ticks
    ax1.set_yticks([])
    plt.tight_layout()

interval = duration*1e3/steps
animation = FuncAnimation(fig, update, frames=steps, interval=interval)
plt.show()