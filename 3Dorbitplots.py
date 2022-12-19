import numpy as np
import matplotlib.pyplot as plt

def euler_solve3(M,G,r0,v0,dt,tf):
    
    time = np.arange(0,tf,dt)

    r = np.zeros((len(time),3))
    r[0] = r0

    v = np.zeros((len(time),3))
    v[0] = v0

    for i in range(len(time)-1):
        v[i+1] = v[i] - (dt*M*G*r[i])/(np.linalg.norm(r[i])**3)
        
        r[i+1] = r[i] + (dt*v[i])
    
    return time,r,v

def verlet_solve3(M,G,r0,v0,dt,tf):

    time = np.arange(0, tf, dt)

    r = np.zeros((len(time),3))
    r[0] = r0

    v = np.zeros((len(time),3))
    v[0] = v0

    r[1] = r[0] + (dt*v[0])

    for i in range(1, len(time)-1):
        
        r[i+1] = 2*r[i] - r[i-1] - ((dt**2)*G*M*r[i])/(np.linalg.norm(r[i])**3)
        
        v[i] = (r[i+1] - r[i-1])/(2*dt)

    v[-1] = v[-2] - (dt*G*M*r[-2])/(np.linalg.norm(r[i])**3)

    return time,r,v


time,r,v = verlet_solve3(6.39e23,6.674e-11,[800e3+3389.5e3,1000e3,4000e3],[300,900,1000],0.05,15000)


x_positions = [r[i][0] for i in range(len(r))]
y_positions = [r[i][1] for i in range(len(r))]
z_positions = [r[i][2] for i in range(len(r))]

ax = plt.figure().add_subplot(projection='3d')

ax.plot(x_positions, y_positions, z_positions)
ax.legend()

plt.show()
