from util import sensor_positions, dipole_field, add_noise, residuals
from scipy.optimize import least_squares

pos_iman = [0.9, 0.0, 0.2]  # Posición real del imán
# Campo teórico en los sensores	
B = []
for sensor_pos in sensor_positions:
    b = dipole_field(pos_iman, sensor_pos)
    B.append(b)

# Campo medido en los sensores (con ruido)
measured_B = add_noise(B)

p0 = [0.0, 0.0, 0.1]  # Suposición inicial
result = least_squares(residuals, p0, args=(sensor_positions, measured_B), method='lm',
                       ftol=1e-15, xtol=1e-12, gtol=1e-12, max_nfev=1000000)
estimated_position = result.x
print("Posición estimada del imán:", estimated_position)
print("Diferencia posiciones:", pos_iman - estimated_position)