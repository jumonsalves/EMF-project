from util import dipole_field
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

cantidad = 8
limite = 3
max_angle = 360
fps = 60
dpi = 300
file_name = "magnetic_field"

x = np.linspace(-limite, limite, cantidad)
y = np.linspace(-limite, limite, cantidad)
z = np.linspace(-limite, limite, cantidad)


posiciones = []
for xt in x:
    for yt in y:
        for zt in z:
            posiciones.append([xt, yt, zt])

B = []
for position in posiciones:
    B.append(dipole_field([0, 0, 0], position, G=[0,0,0]))  # Convertir a microteslas
    

b_array = np.array(B)
B_mag = []
for i in b_array:
    B_mag.append(np.linalg.norm(i))


fig = plt.figure()
ax = plt.axes(projection="3d")
writer = FFMpegWriter(fps=fps)
with writer.saving(fig, f"plots/{file_name}.mp4", dpi):
    for angle in range(0, max_angle):
        ax.view_init(20, angle)
        for i, position in enumerate(posiciones):
            ax.quiver(
                position[0],
                position[1],
                position[2],
                B[i][0],
                B[i][1],
                B[i][2],
                normalize=True,
                alpha=np.abs(1/(4+np.log10(B_mag[i]))),
                color="b",
                label=("Campo magnético" if B_mag[i] == np.max(B_mag) else None)
            )
        ax.scatter(0, 0, 0, marker=".", color="r", label="Imán")
        ax.set_xlabel("Eje X")
        ax.set_ylabel("Eje Y")
        ax.set_zlabel("Eje Z")
        ax.legend()
        writer.grab_frame()
        plt.cla()
        