from util import dipole_field, add_noise, residuals, goal_condition, sensor_positions
import numpy as np
from scipy.optimize import least_squares

def y_trajectory(x):
    return -(((x - 0.93) / 2.8) ** 2) + 0.24
def z_trajectory(x):
    return 0.5 - x / 5.25

x_vals = np.linspace(-0.3, 2.1, 200)
y_vals = [y_trajectory(x) for x in x_vals]
z_vals = [z_trajectory(x) for x in x_vals]

p0 = [0.9, 0.0, 0.1]  # Suposición inicial
model_positions = []
for i, x in enumerate(x_vals):
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
    result = least_squares(residuals, p0,
                           args=(sensor_positions, measured_B), method='lm',
                           ftol=1e-15, xtol=1e-12, gtol=1e-12, max_nfev=1000000)
    estimated_position = result.x
    print("Disco en el area de gol ✅" if goal_condition(estimated_position)
          else "Disco fuera del area de gol ❌")
    model_positions.append(estimated_position)
    p0 = estimated_position
