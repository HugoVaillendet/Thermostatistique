import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#On définit les paramètres de la simulation

N=100 #Nombre de particules
mass=1 #Masse des particules
radius=0.1 #rayon des particules
L=20 #Longueur de la boite
v_0=2 #Norme des vitesses initiales
duration=10 #Durée de la simulation
dt=0.01 #Intervale de temps
steps=int(duration/dt) #Nombre de pas

#La fonction meshCreation crée un réseau régulier de particules dans la boit de longueur L.
def meshCreation(N, L, radius):
    grid_points=int(np.ceil(np.sqrt(N))) #On crée les paramètres de la grille. Prendre l'arrondi de la racine de N assure que (N+1)>=grid^2>=N.
    space=L/grid_points #On calcule l'espacement de chaque particules.
    x=np.linspace(radius+space/2, L-radius - space/2, grid_points) #On crée une rangée de particules

    X_mat,Y_mat=np.meshgrid(x,x, indexing='ij') #On crée à l'aide de la fonction meshgrid de NumPy une matrice des positon pour chaque coordonnées (x, y).
    x, y = X_mat.ravel(), Y_mat.ravel() #On convertit les matrices en array NumPy 1D.
    position=np.stack((x, y)).T #On stack et transpose les deux array de coordonnées de manière à obtenir un array contenant la liste des coordonnées x et y pour un maillage grid^2.

    return position[:N] #On retourne un slice jusque à N de l'array position pour n'avoir que N particules (grid^2>N).

#La fonction velocityInit crée un tableau des composantes x et y des vitesses initiales des particules.
def velocityInit(N):
    theta=np.random.uniform(0, 2*np.pi, size=N) #On crée un array 1D de taille N contenant des valeur aléatoires de 0 à 2pi.
    v_x,v_y=v_0*np.cos(theta), v_0*np.sin(theta) #On calcule les composante (v_x, v_y) des vecteurs v_n d'angles relatif à la particule aléatoires et de norme fixe v_0.
    velocity=np.stack((v_x,v_y), axis=1) #On stack ces valeurs dans un seul array 2D pour chaque composantes.
    return velocity

def collisions(position, velocity, radius, L, dt, N):
    position_next = position + velocity * dt #On calcule l'array position pour (t+dt).

    #Les arrays velocity position et position_next étant tous trois des arrays (N,2) on peux utiliser le boolean masking ici.

    velocity[position_next[:, 0] < radius, 0] *= -1
    """
    Cette ligne de code ci dessus applique la collision élastique des particules (voir section plus sur les collisions) au mur de gauche
    Les éléments (N,0) de velocity pour lesquels position_next[:, 0] (c'est à dire l'ensemble des coordonnées x de position_next)
    sont inférieur à 1 radius se voient appliquer l'opération *(-1).
    C'est équivalent à une boucle for qui parcours les éléments (N,0) de velocitiy (c'est à dire pour ce cas les éléments orthogonaux) et qui applique la
    condition if (positon_next[:, 0] < radius).
    Néanmoins la vectorisation avec NumPy permet des gains de performance.
    """
    velocity[position_next[:, 0] > L - radius, 0] *= -1 #Code identique pour le mur de droite
    velocity[position_next[:, 1] < radius, 1] *= -1 #Code identique pour le mur du bas
    velocity[position_next[:, 1] > L - radius, 1] *= -1 #Code identique pour le mur du haut

    for i in range(N):
        for j in range(i + 1, N):
            """
            On cherche à calculer quelles particules pourraient entrer en collisions entre elles à un instant t.
            Pour se faire on parcours d'abord les N particules avec un boucle for.
            Pour chacune des particules on teste les N-(i+1) particules.
            En effet pas besoin de tester les N-1 particules. Si on prend la première particule P1 on teste N-1 particules (P1<->P2; P1<->P3; ...;P1<->P(N-1)).
            Quand on teste la deuxième, inutile de tester la paire P1 et P2 car ce calcul a déjà eu lieu lors du test pour P1.
            """
            if np.linalg.norm(position_next[i] - position_next[j]) < 2 * radius: #On teste si la distance entre les deux particules considérée est < 2 radius.

                # On applique les formules détailées dans la section collision (ici les masses sont identiques donc s'annulent)

                delta_velocity = velocity[i] - velocity[j]
                delta_r = position[i] - position[j]

                velocity[i] -= delta_r.dot(delta_velocity) / (delta_r.dot(delta_r)) * delta_r
                velocity[j] += delta_r.dot(delta_velocity) / (delta_r.dot(delta_r)) * delta_r

def step(position, velocity, radius, L, dt, N):
    collisions(position, velocity, radius, L, dt, N) #On calcule les collision pour un step.
    position += velocity * dt #On met à jour l'array position.
    #Redondance

def simulation(position, velocity, radius, L, dt, N, steps):
    positions = np.zeros((steps, N, 2)) #On crée un array positions qui stocke l'entièreté des positons pour chaque pas de temps
    velocities = np.zeros((steps, N))#On crée un array velocities qui stocke l'entièreté des vitesses pour chaque pas de temps.

    for i in range(steps): #On parcours chaques step.
        step(position, velocity, radius, L, dt, N) #On applique la fonction step qui calcule nos postions et vitesses
        positions[i, :, :] = position #On place l'entièreté de l'array position dans l'array 3D positions.
        velocities[i, :] = np.linalg.norm(velocity, axis=1) # On place la norme des vitesses dans l'array 2D velocities.

    return positions, velocities

def update(frame):
    axis_1.clear() #On clear axis_1.
    axis_2.clear() #On clear axis_1.
    for i in range(N): #On dessine les N points points.
        x, y = positions[frame,i, 0], positions[frame, i, 1]
        circle = plt.Circle((x, y), radius, fill=True)
        axis_1.add_artist(circle)
        
    axis_2.hist(velocities[frame], density=True, bins=25)

    #On met en forme les axes et le titre du graphique.
    axis_1.set_xlabel('$X$', fontsize=15)
    axis_1.set_ylabel('$Y$', fontsize=15)
    axis_1.set_title("Animation d'un gaz parfait", fontsize=15)
    axis_1.set_xlim(0, L)
    axis_1.set_ylim(0, L)
    axis_1.set_aspect('equal')
    axis_1.set_xticks([])
    axis_1.set_yticks([])

    axis_2.set_xlim(0, 15)
    axis_2.set_ylim(0, 6.5)
    axis_2.set_xlabel('$v$ (m/s)', fontsize=15)
    axis_2.set_ylabel('frequency', fontsize=15)
    axis_2.set_title('Distribution des vitesses', fontsize=15)
    axis_2.plot(v,f, label = 'Distribution Maxwell-Boltzmann')
    axis_2.legend(fontsize=15)

    plt.tight_layout()


def MaxwellBoltzmann(v_0, mass,v):
    E_c_avrg=0.5*mass*(v_0**2)
    TkB=E_c_avrg
    sigma=TkB/mass
    f=(v/sigma)/np.exp(-v**2/(2*sigma))
    return f

particles_position=meshCreation(N, L, radius) #On initialise le réseau de particules.

particle_velocity=velocityInit(N) #On initialise les vitesses des particules du réseau.

v=np.linspace(0, 30, 600)
f=MaxwellBoltzmann(v_0, mass,v) #

#On affiche le réseau de particules :
fig=plt.figure()
axis_1=fig.add_subplot(111)
for(x,y) in particles_position:
    axis_1.add_artist(plt.Circle((x,y), radius, color='r'))
axis_1.set_xlim(0, L)
axis_1.set_ylim(0, L)
axis_1.set_aspect('equal')
plt.show()

#On simule pour chaque pas de temps les collisions :
positions, velocities = simulation(particles_position, particle_velocity, radius, L, dt, N, steps)

#On procède à l'animation des particules.
fig,(axis_1,axis_2) = plt.subplots(1,2, figsize = (12,6))

interval = duration*1e3/steps
animation = FuncAnimation(fig, update, frames=steps, interval=interval)
plt.show()