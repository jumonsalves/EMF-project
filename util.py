import numpy as np
np.random.seed(0)

def dipole_field(magnet_pos, sensor_pos, G=None):
    mu_0 = 4 * np.pi * 1e-7  # Permeabilidad magnética del vacío
    magnetic_moment = 0.26  # Momento magnético del imán
    dipole_weight = (mu_0 / (4 * np.pi)) * magnetic_moment  # Peso del dipolo magnético
    dipole_weight = [0.0, 0.0, dipole_weight]  # Peso del dipolo magnético (apuntando en la dirección z)
    
    if G is None:
        G = np.array([26582e-9, -3971e-9, 13422e-9])
    else:
        G = np.array(G, dtype=float)

    # Asegurarse de que las posiciones y la orientación sean arrays de numpy de tipo float
    magnet_pos = np.array(magnet_pos, dtype=float)
    sensor_pos = np.array(sensor_pos, dtype=float)

    # Calcular el vector r desde el imán hasta el sensor
    r = sensor_pos - magnet_pos
    r_norm = np.linalg.norm(r)

    if r_norm == 0:
        raise ValueError("La posición del sensor coincide con la del imán (r = 0).")
    # Calcular el campo magnético usando el modelo dipolar:
    B = G + (
        (3 * r * np.dot(dipole_weight, r) / (r_norm**5))
          - (dipole_weight / (r_norm**3))
    )

    return B

def add_noise(B:list):
    B_rand = []
    for b in B:
        B_rand.append(np.array(b) + np.random.rand(3) * 1e-7)
    return np.array(B_rand)

def residuals(p, sensor_positions, measured_B):
    # p es el vector [x, y, z] de la posición del imán.
    errors = []
    for sensor_pos, B_meas in zip(sensor_positions, measured_B):
        # Calculamos el campo predicho usando la función dipole_field:
        B_pred = dipole_field(p, sensor_pos)
        # La diferencia (residuo) entre el predicho y el medido:
        errors.append(B_pred - B_meas)
    # Concatenamos todos los residuos en un vector unidimensional:
    return np.concatenate(errors)

def goal_condition(pos):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    return (y>0.18+0.038) and (x>0.0) and (x<1.8) and (z>0.0) and (z<1.2)

d = 0.36
sensor_positions = [
    [-1.0, -0.5, 0],
    [-0.5, -0.5, 0],
    [0.0, -0.5, 0],
    [0.5, -0.5, 0],
    [1.0, -0.5, 0],
    [1.5, -0.5, 0],
    [2.0, -0.5, 0],
    [2.5, -0.5, 0],
    [3.0, -0.5, 0],
    [3.5, -0.5, 0],
    [4.0, -0.5, 0],
    [4.5, -0.5, 0],
    [5.0, -0.5, 0],
    [5.5, -0.5, 0],
    [6.0, -0.5, 0],
    [-1.0, 0.0, 0],
    [-0.5, 0.0, 0],
    [0.0, 0.0, 0],
    [0.5, 0.0, 0],
    [1.0, 0.0, 0],
    [1.5, 0.0, 0],
    [2.0, 0.0, 0],
    [2.5, 0.0, 0],
    [3.0, 0.0, 0],
    [3.5, 0.0, 0],
    [4.0, 0.0, 0],
    [4.5, 0.0, 0],
    [5.0, 0.0, 0],
    [5.5, 0.0, 0],
    [6.0, 0.0, 0],
    [-1.0, 0.5, 0],
    [-0.5, 0.5, 0],
    [0.0, 0.5, 0],
    [0.5, 0.5, 0],
    [1.0, 0.5, 0],
    [1.5, 0.5, 0],
    [2.0, 0.5, 0],
    [2.5, 0.5, 0],
    [3.0, 0.5, 0],
    [3.5, 0.5, 0],
    [4.0, 0.5, 0],
    [4.5, 0.5, 0],
    [5.0, 0.5, 0],
    [5.5, 0.5, 0],
    [6.0, 0.5, 0],
    [-1.0, 1.0, 0],
    [-0.5, 1.0, 0],
    [0.0, 1.0, 0],
    [0.5, 1.0, 0],
    [1.0, 1.0, 0],
    [1.5, 1.0, 0],
    [2.0, 1.0, 0],
    [2.5, 1.0, 0],
    [3.0, 1.0, 0],
    [3.5, 1.0, 0],
    [4.0, 1.0, 0],
    [4.5, 1.0, 0],
    [5.0, 1.0, 0],
    [5.5, 1.0, 0],
    [6.0, 1.0, 0],
]
sensor_amount = len(sensor_positions)
for i in range(sensor_amount):
    sensor_positions[i] = np.array(sensor_positions[i], dtype=float)
    sensor_positions[i] *= d

x_coords = [p[0] for p in sensor_positions]
y_coords = [p[1] for p in sensor_positions]
z_coords = [p[2] for p in sensor_positions]
sensor_cords = [x_coords, y_coords, z_coords]
