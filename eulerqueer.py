import numpy as np
import matplotlib.pyplot as plt

g = 9.81
m = 1.0
rho = 1.225
A = 0.01
C_d = 0.3
C_L = 0.5
v_wind = np.array([2.0, 0.0, 0.0])

v0 = np.array([30.0, 0.0, 20.0])
R0 = np.array([0.0, 0.0, 0.0])

dt = 0.01
t_init = 0.0
t_end = 6.0

theta_schedule = {"times_s": [0.0, t_end], "values_deg": [0.0, 10.0]}
phi_schedule = {"times_s": [0.0, t_end], "values_deg": [45.0, 0.0]}
gamma_schedule = {"times_s": [0.0, t_end], "values_deg": [0.0, 0.0]}


def interp_angle(schedule: dict, t: float) -> float:
    return np.interp(t, schedule["times_s"], schedule["values_deg"]) * np.pi / 180.0


def net_force(t: float, v_vec: np.ndarray) -> np.ndarray:
    theta = interp_angle(theta_schedule, t)
    phi = interp_angle(phi_schedule, t)
    gamma = interp_angle(gamma_schedule, t)

    v_eff = v_vec + v_wind
    v_sq = float(np.dot(v_eff, v_eff))

    drag = 0.5 * C_d * rho * A * v_sq
    lift = 0.5 * C_L * rho * A * v_sq
    thrust_minus_drag = -drag

    Fx = thrust_minus_drag * np.cos(theta) * np.cos(phi) + lift * np.sin(phi) * np.cos(gamma)
    Fy = thrust_minus_drag * np.sin(theta) * np.cos(phi) + lift * np.cos(phi) * np.sin(gamma)
    Fz = (
        thrust_minus_drag * np.cos(theta) * np.sin(phi)
        + lift * np.cos(phi) * np.cos(gamma)
        - m * g
    )
    return np.array([Fx, Fy, Fz], dtype=float)


def simulate():
    n = int((t_end - t_init) / dt)
    t_arr = np.zeros(n + 1, dtype=float)
    v_arr = np.zeros((n + 1, 3), dtype=float)
    R_arr = np.zeros((n + 1, 3), dtype=float)

    t_arr[0] = t_init
    v_arr[0] = v0
    R_arr[0] = R0

    for i in range(1, n + 1):
        t_arr[i] = t_arr[i - 1] + dt

        F = net_force(t_arr[i - 1], v_arr[i - 1])
        v_arr[i] = v_arr[i - 1] + (F / m) * dt
        R_arr[i] = R_arr[i - 1] + 0.5 * (v_arr[i - 1] + v_arr[i]) * dt

        if (i > 0) and (R_arr[i, 2] < 0.0):
            t_arr = t_arr[: i + 1]
            v_arr = v_arr[: i + 1]
            R_arr = R_arr[: i + 1]
            break

    return t_arr, v_arr, R_arr


t_arr, v_arr, R_arr = simulate()

fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax1.plot(R_arr[:, 0], R_arr[:, 1], R_arr[:, 2], "b-", linewidth=1.5)
ax1.scatter(*R_arr[0], color="green", s=60, label="Launch")
ax1.scatter(*R_arr[-1], color="red", s=60, label="Impact")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
ax1.legend()

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(R_arr[:, 0], R_arr[:, 2], "b-", linewidth=1.5)
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Z (m)")

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(t_arr, v_arr[:, 0], label="vx")
ax3.plot(t_arr, v_arr[:, 1], label="vy")
ax3.plot(t_arr, v_arr[:, 2], label="vz")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Velocity (m/s)")
ax3.legend()

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(t_arr, np.linalg.norm(v_arr, axis=1), "k-", linewidth=1.5)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Speed (m/s)")

plt.tight_layout()
plt.savefig("trajectory_3d.png", dpi=150)
plt.show()