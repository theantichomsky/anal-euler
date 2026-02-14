import numpy as np
import matplotlib.pyplot as plt

g = 9.81
m = 1.0         # mass in kg
v_init = 50.0   # init upward velocity in m/s
R_init = 0.0    # init position in m

dt = 0.1
t_init = 0
t_end = 10
n = int((t_end - t_init) / dt)

t = np.zeros(n + 1)
v = np.zeros(n + 1)
R = np.zeros(n + 1)

t[0] = t_init
v[0] = v_init
R[0] = R_init

def F(t, v):
    return -m * g

for i in range(1, n + 1):
    t[i] = t[i-1] + dt
    v[i] = v[i-1] + F(t[i-1], v[i-1]) * dt / m
    R[i] = R[i-1] + 0.5 * (v[i-1] + v[i]) * dt

t_exact = np.linspace(t_init, t_end, 500)
v_exact = v_init - g * t_exact
R_exact = R_init + v_init * t_exact - 0.5 * g * t_exact**2

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax1.plot(t, R, 'o', markersize=3, label='Euler position')
ax1.plot(t_exact, R_exact, '--', label='Exact position')
ax1.set_ylabel('Position R (m)')
ax1.legend()

ax2.plot(t, v, 'o', markersize=3, label='Euler velocity')
ax2.plot(t_exact, v_exact, '--', label='Exact velocity')
ax2.set_ylabel('v')
ax2.set_xlabel('t') 
ax2.legend()

plt.tight_layout()
plt.show()

print(f"At t={t_end}s:")
print(f"  R: {R[-1]:.3f} m,  Exact: {R_init + v_init*t_end - 0.5*g*t_end**2:.3f} m")
print(f"  v: {v[-1]:.3f} m/s, Exact: {v_init - g*t_end:.3f} m/s")