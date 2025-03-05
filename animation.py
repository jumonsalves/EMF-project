from util import dipole_field, add_noise, residuals, goal_condition, sensor_positions, sensor_cords
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

fig = plt.figure()
ax = plt.axes(projection="3d")
fps = 60
seconds = 2
dpi = 300
file_name = "anim"


def y_trajectory(x):
    return -(((x - 0.93) / 2.8) ** 2) + 0.24
def z_trajectory(x):
    return 0.5 - x / 5.25

x_vals = np.linspace(-0.3, 2.1, fps * seconds)
y_vals = [y_trajectory(x) for x in x_vals]
z_vals = [z_trajectory(x) for x in x_vals]

writer = FFMpegWriter(fps=fps)
with writer.saving(fig, f"plots/{file_name}.mp4", dpi):
    p0 = [0.9, 0.0, 0.2]  # Suposición inicial
    model_positions = []
    for i, x in enumerate(x_vals):
        # Limite de los ejes
        ax.set_xlim3d(-0.36, 2.16)
        ax.set_ylim3d(-0.5, 0.36 + 0.5)
        ax.set_zlim3d(-0.00, 1.4)
        # Etiquetas de los ejes
        ax.set_xlabel("Eje X")
        ax.set_ylabel("Eje Y")
        ax.set_zlabel("Eje Z")
        # Portetia
        ax.plot([0, 1.8], [0.18, 0.18], [0, 0], color="gray")
        ax.plot([0.0, 1.8], [0.18, 0.18], [1.2, 1.2], color="red", linewidth=4.0)
        ax.plot([0.0, 0.0], [0.18, 0.18], [0.0, 1.2], color="red", linewidth=4.0)
        ax.plot([1.8, 1.8], [0.18, 0.18], [0.0, 1.2], color="red", linewidth=4.0)

        # Posición del imán
        magnet_pos = [x, y_vals[i], z_vals[i]]
        # Campo teórico en los sensores	
        B = []
        for sensor_pos in sensor_positions:
            b = dipole_field(magnet_pos, sensor_pos)
            B.append(b)

        # Campo medido en los sensores (con ruido)
        measured_B = add_noise(B)
        # Regresión no lineal
        result = least_squares(residuals, p0, args=(sensor_positions, measured_B), method='lm', ftol=1e-15, xtol=1e-12, gtol=1e-12, max_nfev=1000000)
        estimated_position = result.x
        model_positions.append(estimated_position)
        p0 = estimated_position

        real_color = "green" if goal_condition(magnet_pos) else "red"
        estimated_color = "green" if goal_condition(estimated_position) else "red"
        # Dibujar los sensores
        ax.scatter(*sensor_cords, marker=".", color="blue")
        # Dibujar la posición real el imán
        ax.scatter(*magnet_pos, color=real_color, marker="o")
        # Trayectoria real del imán
        ax.plot(
            x_vals[0:i],
            y_vals[0:i],
            z_vals[0:i],
            color=real_color,
            label="Trayectoria real",
        )
        # Vector que apunta al imán 
        # ax.quiver(magnet_pos[0], magnet_pos[1], 0, 0, 0, magnet_pos[2], color=real_color)  

        # Dibujar la posición estimada del imán
        ax.scatter(*estimated_position, color=estimated_color, marker="o", alpha=0.5, label=("GOL DETECTADO" if goal_condition(estimated_position) else None))
        # Trayectoria estimada del imán
        ax.plot(
            [pos[0] for pos in model_positions],
            [pos[1] for pos in model_positions],
            [pos[2] for pos in model_positions],
            color=estimated_color,
            alpha=0.5,
            label="Trayectria estimada",
        )
        # ax.plot(
        #     [pos[0] for pos in model_positions],
        #     [pos[1] for pos in model_positions],
        #     0,
        #     color=estimated_color,
        #     alpha=0.5,
        # )
        # Vector que apunta a la posoción estimada del imán
        # ax.quiver(estimated_position[0], estimated_position[1], 0, 0, 0, estimated_position[2], color=estimated_color, alpha=0.5)  

        # Dibujar el campo magnético en los sensores * 1e4 / 2
        for j, position in enumerate(sensor_positions):
            ax.quiver(
                position[0],
                position[1],
                position[2],
                measured_B[j][0]*1e4/2,
                measured_B[j][1]*1e4/2,
                measured_B[j][2]*1e4/2,
                alpha= 0.5,
                color="b",
            )
        
        ax.legend()
        # ax.view_init(5, 0)
        writer.grab_frame()
        plt.cla()
        # Progreso en porcentaje
        print(f"{i/len(x_vals)*100:.2f}%")
     
