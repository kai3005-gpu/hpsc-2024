import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

NX = 41
NY = 41

def main():
    x = np.linspace(0, 2, NX)
    y = np.linspace(0, 2, NY)
    X, Y = np.meshgrid(x, y)

    u = np.zeros((NY, NX))
    v = np.zeros((NY, NX))
    p = np.zeros((NY, NX))

    with open('u.dat', 'r') as f:
        uraw = f.readlines()
    with open('v.dat', 'r') as f:
        vraw = f.readlines()
    with open('p.dat', 'r') as f:
        praw = f.readlines()
    
    frames = []  # To store frames for GIF

    for n in range(len(uraw)):
        plt.figure(figsize=(10, 5))  # Adjust the size as needed
        plt.clf()
        u_flattened = [float(val) for val in uraw[n].strip().split() if val]
        v_flattened = [float(val) for val in vraw[n].strip().split() if val]
        p_flattened = [float(val) for val in praw[n].strip().split() if val]

        for j in range(NY):
            for i in range(NX):
                u[j, i] = u_flattened[j * NX + i]
                v[j, i] = v_flattened[j * NX + i]
                p[j, i] = p_flattened[j * NX + i]

        plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
        plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
        plt.title(f'C++, n = {n}')

        # Save plot to a PIL Image object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img)
        plt.close()

    # Save frames as a GIF
    frames[0].save('fluid_simulation.gif', format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)

if __name__ == '__main__':
    main()
