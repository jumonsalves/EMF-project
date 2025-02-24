import numpy as np


def dipole_field(magnet_pos, sensor_pos, G=None):
    """
    Calcula el campo magnético B = [Bx, By, Bz] en la posición del sensor (sensor_pos)
    generado por un imán ubicado en magnet_pos, con un "peso" dipolar m_bar y orientación m_hat.

    Parámetros:
      magnet_pos : array-like, [x0, y0, z0] (ubicación del imán).
      sensor_pos : array-like, [xs, ys, zs] (ubicación del sensor).
      magnetic_moment : float (momento magnético del imán).
      G          : vector del campo de perturbación (por defecto se considera [0, 0, 0]).

    Retorna:
      B : array, [Bx, By, Bz] (campo magnético en sensor_pos).
    """
    mu_0 = 4 * np.pi * 1e-7  # Permeabilidad magnética del vacío
    magnetic_moment = 0.26  # Momento magnético del imán
    dipole_weight = (mu_0 / (4 * np.pi)) * magnetic_moment  # Peso del dipolo magnético
    dipole_weight = [
        0.0,
        0.0,
        dipole_weight,
    ]  # Peso del dipolo magnético (apuntando en la dirección z)

    if G is None:
        G = np.array([0.0, 0.0, 0.0])
    else:
        G = np.array(G, dtype=float)

    # Asegurarse de que las posiciones y la orientación sean arrays de numpy de tipo float
    magnet_pos = np.array(magnet_pos, dtype=float)
    sensor_pos = np.array(sensor_pos, dtype=float)

    # Calcular el vector r desde el imán hasta el sensor
    r = sensor_pos - magnet_pos
    r_norm = np.linalg.norm(r)

    if r_norm == 0:
        # return np.array([0.0, 0.0, 0.0])
        raise ValueError("La posición del sensor coincide con la del imán (r = 0).")

    # Calcular el campo magnético usando el modelo dipolar:
    B = G + (
        (3 * r * np.dot(dipole_weight, r) / (r_norm**5)) - (dipole_weight / (r_norm**3))
    )

    return B


d = 0.36
# Sensor positions
sensor_positions = [
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, 0],
    [3, 0, 0],
    [4, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [2, 1, 0],
    [3, 1, 0],
    [4, 1, 0],
]
sensor_amount = len(sensor_positions)
for i in range(len(sensor_positions)):
    sensor_positions[i] = np.array(sensor_positions[i], dtype=float)
    sensor_positions[i] *= d

x_coords = [p[0] for p in sensor_positions]
y_coords = [p[1] for p in sensor_positions]
z_coords = [p[2] for p in sensor_positions]
sensor_cords = [x_coords, y_coords, z_coords]
