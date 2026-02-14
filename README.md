# anal-euler

Tool for solving a recursive function v' and then let v'' be a constant velocity vector such that we have some position function s(t), but by superimposing do you mean that s(t + dt) is supposed to be the integrand of v(i) + w? <-- where i is just the time from t to t + dt.

So in other words, s(t+dt) = s(t) + ∫t, t+dt (v(i) + w))di
-> ∫t, t+dt (v(i)+w))di = (v' + w) x dt
-> s(t+dt) = s(t) + (v' + w) x dt

## Run
```console
python3 point_mass_3d.py
```
Simulation stops immediately at ground impact (z ≤ 0)

## Config
There are two scripts including in the repo, one that simulates a projectile in a 1D plane and another that simulates a projectile in a 3D plane. 
All parameters are at the top of the script:
    -Physical: m, rho, A, C_d, C_L
    -Initial conditions: v0, R0
    -Wind: v_wind — constant 3D vector added to airspeed
    -Time: dt, t_init, t_end
    -Angle schedules: theta_schedule, phi_schedule, gamma_schedule — each is a dict with times_s and values_deg arrays, linearly interpolated at each time step

I set thrust to zero, but it can be added to sum_forces(). Also, the position update uses the trapezoidal rule to reduce drift. 
