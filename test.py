import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import PillowWriter
from util import dipole_field, sensor_positions, sensor_cords, sensor_amount

fig = plt.figure()
ax = plt.axes(projection="3d")
writer = PillowWriter(fps=10)
# writer = FFMpegWriter(fps=10)


def trajectory(x):
    return -(((x - 0.83) / 3) ** 2) + 0.21


x_vals = np.linspace(-0.38, 1.82, 50)

# with writer.saving(fig, "test.mp4", 250):
with writer.saving(fig, "test.gif", 250):
    for i, x in enumerate(x_vals):
        # Limite de los ejes
        ax.set_xlim3d(-0.38, 1.82)
        ax.set_ylim3d(-0.5, 0.36 + 0.5)
        ax.set_zlim3d(-0.00, 2)
        # Etiquetas de los ejes
        ax.set_xlabel("Eje X")
        ax.set_ylabel("Eje Y")
        ax.set_zlabel("Eje Z")

        # Posición del imán
        magnet_pos = [x, trajectory(x), 0.5 - x / 4.55]
        # Dibujar los sensores
        ax.scatter(*sensor_cords, marker=".", color="r")
        # Dibujar el imán
        ax.scatter(*magnet_pos, color="black", marker="o")
        # Calcular el campo magnético en cada sensor
        B = [0] * sensor_amount
        B_dir = [0] * sensor_amount
        for j, position in enumerate(sensor_positions):
            B[j] = dipole_field(magnet_pos, position) * 1e6  # Convertir a microteslas
            B_dir[j] = B[j] / np.linalg.norm(B[j]) * 0.1

            # Vecotres con la dirección del campo magnético pero magnitud acentuada
            B[j] = dipole_field(magnet_pos, position) * 1e4 / 15 + B_dir[j] / 0.5
        # Dibujar el campo magnético en los sensores
        for k, position in enumerate(sensor_positions):
            ax.quiver(
                position[0],
                position[1],
                position[2],
                B[k][0],
                B[k][1],
                B[k][2],
            )

        ax.quiver(
            x, trajectory(x), 0, 0, 0, 0.5 - x / 4.55, color="black", alpha=0.5
        )  # Vector que apunta al imán

        # Proyección de la trayectoria del imán en el plano XY
        ax.plot(
            x_vals[0:i],
            trajectory(x_vals[0:i]),
            np.zeros(len(x_vals[0:i])),
            color="black",
            alpha=0.5,
        )
        ax.plot(
            x_vals[0:i],
            trajectory(x_vals[0:i]),
            0.5 - x_vals[0:i] / 4.55,
            color="slategray",
        )

        # Porteria
        ax.plot([-0.18, 1.62], [0.18, 0.18], [1.2, 1.2], color="black", linewidth=4.0)
        ax.plot([-0.18, -0.18], [0.18, 0.18], [0.0, 1.2], color="black", linewidth=4.0)
        ax.plot([1.62, 1.62], [0.18, 0.18], [0.0, 1.2], color="black", linewidth=4.0)
        ax.plot([-0.18, 1.62], [0.18, 0.18], [0, 0], color="red")
        # plt.title(f"Posición del imán: ({x:.2f}, {trajectory(x):.2f}, {0.5 - x / 4.55:.2f})")

        writer.grab_frame()
        plt.cla()
