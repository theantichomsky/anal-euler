# Euler's Approximation
Tool for solving a recursive function v' and then let v'' be a constant velocity vector such that we have some position function s(t) st
s(t+dt) = s(t) + ∫t, t+dt (v(i) + w))di
-> ∫t, t+dt (v(i)+w))di = (v' + w) x dt
-> s(t+dt) = s(t) + (v' + w) x dt

## Run
```console
python3 [filename].py
```

Projectile scripts stop at ground impact (z ≤ 0). okbuddyeuler.py runs through all five flight phases and stops at the destination vertiport, which are predetermined as the project is designed. 

## Config

There are three scripts in the repo: one that simulates a projectile in a 1D plane, another in a 3D plane, and a third (okbuddyeuler.py) that simulates a full eVTOL mission profile between vertiport nodes.

For eulerprojectile.py and eulertest1.py, all parameters are at the top of the script:

    -Physical: m, rho, A, C_d, C_L
    
    -Initial conditions: v0, R0
    
    -Wind: v_wind — constant 3D vector added to airspeed
    
    -Time: dt, t_init, t_end
    
    -Angle schedules: theta_schedule, phi_schedule, gamma_schedule — each is a dict with times_s and values_deg arrays, linearly interpolated at each time step
    

For okbuddyeuler.py, the parameters are also at the top of the script:


    -Aircraft: m, rho, A, C_d, C_L
    
    -Power: power_hover, power_climb, power_cruise, power_descent, prop_eff
    
    -Battery: battery_kwh, reserve_min
    
    -Flight profile: v_cruise, v_climb_h, climb_rate, descent_rate, cruise_alt_agl, hover_alt_agl
    
    -Wind: v_wind — constant 3D vector, applied only in the position step
    
    -Route: origin_id, dest_id — keys into the nodes dict


eulerprojectile.py / eulertest1.py: Thrust is set to zero but can be added to sum_forces(). Position update uses the trapezoidal rule to reduce drift.
okbuddyeuler.py: Uses angle-based force decomposition (θ, φ, γ) with thrust active across five flight phases. Position update uses forward Euler. Outputs three plots: mission_result.png, mission_3d.png, mission_map.png.
