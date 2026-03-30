import numpy as np
import matplotlib.pyplot as plt

g = 9.81
m = 2177.0
rho = 1.225
A = 12.0
C_d = 0.035
C_L = 0.45

power_hover = 411.0
power_climb = 339.0
power_cruise = 120.0
power_descent = 15.0
prop_eff = 0.85

battery_kwh = 160.0
reserve_min = 20.0

v_cruise = 58.0
v_climb_h = 45.0
climb_rate = 8.0
descent_rate = 5.0

v_wind = np.array([3.0, 0.0, 0.0], dtype=float)

nodes = {
    "UCD":  {"name": "UC Davis",       "lat": 38.5449, "lon": -121.7405, "elev": 16.0},
    "UCB":  {"name": "UC Berkeley",    "lat": 37.8716, "lon": -122.2727, "elev": 54.0},
    "KNUQ": {"name": "Moffett Field",  "lat": 37.4161, "lon": -122.0490, "elev": 10.0},
    "UCM":  {"name": "UC Merced",      "lat": 37.3660, "lon": -120.4248, "elev": 104.0},
    "UCSC": {"name": "UC Santa Cruz",  "lat": 36.9914, "lon": -122.0585, "elev": 230.0},
}

origin_id = "UCD"
dest_id = "UCB"
cruise_alt_agl = 450.0
hover_alt_agl = 50.0

R_EARTH = 6_371_000.0

HOVER_UP = 0
CLIMB = 1
CRUISE = 2
DESCENT = 3
HOVER_DOWN = 4
COMPLETE = 5

phase_names = ["HOVER \u2191", "CLIMB", "CRUISE", "DESCENT", "HOVER \u2193", "DONE"]
phase_colors = {HOVER_UP: "red", CLIMB: "orange", CRUISE: "royalblue",
                DESCENT: "limegreen", HOVER_DOWN: "purple", COMPLETE: "gray"}
phase_power = [power_hover, power_climb, power_cruise, power_descent, power_hover, 0.0]


def geodetic_to_local(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    lat0_r, lon0_r = np.radians(lat0), np.radians(lon0)
    x = R_EARTH * (lon_r - lon0_r) * np.cos(lat0_r)
    y = R_EARTH * (lat_r - lat0_r)
    return float(x), float(y)


orig = nodes[origin_id]
dest = nodes[dest_id]
x_dest, y_dest = geodetic_to_local(dest["lat"], dest["lon"], orig["lat"], orig["lon"])
bearing_xy = np.array([x_dest, y_dest], dtype=float)
bearing_xy /= max(np.linalg.norm(bearing_xy), 1e-9)
bearing = np.array([bearing_xy[0], bearing_xy[1], 0.0], dtype=float)

z_origin = orig["elev"]
z_dest = dest["elev"]
z_transition = z_origin + hover_alt_agl
z_cruise = z_origin + cruise_alt_agl
z_desc_end = z_dest + hover_alt_agl


def get_phase(pos: np.ndarray, phase: int, x_dest: float, y_dest: float) -> int:
    remaining = np.array([x_dest - pos[0], y_dest - pos[1]])
    dist_along = np.dot(remaining, bearing[:2])

    alt_to_descend = z_cruise - z_desc_end
    desc_dist = (alt_to_descend / max(descent_rate, 0.1)) * v_cruise * 0.75

    if phase == HOVER_UP and pos[2] >= z_transition:
        return CLIMB
    if phase == CLIMB and pos[2] >= z_cruise:
        return CRUISE
    if phase == CRUISE and dist_along <= desc_dist + 200:
        return DESCENT
    if phase == DESCENT and pos[2] <= z_desc_end:
        return HOVER_DOWN
    if phase == HOVER_DOWN and pos[2] <= z_dest + 0.5:
        return COMPLETE
    return phase


def get_target_velocity(phase: int, pos: np.ndarray) -> np.ndarray:
    if phase == HOVER_UP:
        return np.array([0.0, 0.0, 4.0])
    if phase == CLIMB:
        climb_frac = min(1.0, (pos[2] - z_transition) / max(z_cruise - z_transition, 1.0))
        h_speed = v_climb_h * min(1.0, climb_frac * 3 + 0.1)
        return np.array([bearing[0] * h_speed, bearing[1] * h_speed, climb_rate])
    if phase == CRUISE:
        return bearing * v_cruise
    if phase == DESCENT:
        h_speed = v_cruise * 0.75
        return np.array([bearing[0] * h_speed, bearing[1] * h_speed, -descent_rate])
    if phase == HOVER_DOWN:
        return np.array([0.0, 0.0, -3.0])
    return np.zeros(3)


def net_force(v_vec: np.ndarray, phase: int, gamma: float = 0.0) -> np.ndarray:
    """Angle-based force decomposition per Citris Sim theory.

    Forces are computed from v_vec (airframe velocity, WITHOUT wind).
    θ (heading) and φ (pitch) are derived from v_vec; γ (roll) is a
    control input (0 for straight-line, non-zero for banking turns).

    ΣF_x = (T − C_d·½ρAv²)·cos θ·cos φ  +  C_L·½ρAv²·sin φ·cos γ
    ΣF_y = (T − C_d·½ρAv²)·sin θ·cos φ  +  C_L·½ρAv²·cos φ·sin γ
    ΣF_z = (T − C_d·½ρAv²)·cos θ·sin φ  +  C_L·½ρAv²·cos φ·cos γ − mg
    """
    v_mag = np.linalg.norm(v_vec)
    v_sq = v_mag ** 2
    v_horiz = np.hypot(v_vec[0], v_vec[1])

    if v_mag > 0.5:
        theta = np.arctan2(v_vec[1], v_vec[0])
        phi = np.arctan2(v_vec[2], v_horiz)
    elif phase in (HOVER_UP, HOVER_DOWN):
        theta = 0.0
        phi = np.pi / 2
    else:
        theta = np.arctan2(bearing[1], bearing[0])
        phi = 0.0

    if phase in (HOVER_UP, HOVER_DOWN):
        T_mag = m * g
        if phase == HOVER_UP:
            T_mag += m * 2.5
    elif phase in (CLIMB, CRUISE, DESCENT):
        power_w = phase_power[phase] * 1000.0 * prop_eff
        T_mag = power_w / v_mag if v_mag > 1.0 else 0.0
    else:
        T_mag = 0.0

    q_v2 = 0.5 * rho * A * v_sq

    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    cg, sg = np.cos(gamma), np.sin(gamma)

    thrust_drag = T_mag - C_d * q_v2

    F_x = thrust_drag * ct * cp + C_L * q_v2 * sp * cg
    F_y = thrust_drag * st * cp + C_L * q_v2 * cp * sg
    F_z = thrust_drag * ct * sp + C_L * q_v2 * cp * cg - m * g

    return np.array([F_x, F_y, F_z])


def simulate():
    dt_sim = 0.05
    max_steps = int(3600 / dt_sim)

    t_arr = np.zeros(max_steps, dtype=float)
    v_arr = np.zeros((max_steps, 3), dtype=float)
    R_arr = np.zeros((max_steps, 3), dtype=float)
    phase_arr = np.zeros(max_steps, dtype=int)
    power_arr = np.zeros(max_steps, dtype=float)

    R_arr[0] = np.array([0.0, 0.0, z_origin], dtype=float)
    v_arr[0] = np.zeros(3, dtype=float)
    phase_arr[0] = HOVER_UP
    power_arr[0] = phase_power[HOVER_UP] * 1.25

    energy_kwh = 0.0
    step = 0

    for i in range(1, max_steps):
        t_arr[i] = t_arr[i - 1] + dt_sim
        phase = get_phase(R_arr[i - 1], phase_arr[i - 1], x_dest, y_dest)

        if phase == COMPLETE:
            R_arr[i] = R_arr[i - 1]
            v_arr[i] = 0.0
            phase_arr[i] = COMPLETE
            power_arr[i] = 0.0
            step = i
            break

        # ── Theory Step 1: F(t, θ, φ, γ) from v(t) [no wind] ──
        F = net_force(v_arr[i - 1], phase)

        # ── Theory Step 2: v(t+Δt) = v(t) + F(t, θ, φ, γ) × Δt/m ──
        v_physics = v_arr[i - 1] + (F / m) * dt_sim

        # Flight-control overlay: blend toward phase target velocity.
        # (To be replaced by weight-scored navigation in a future revision.)
        v_target = get_target_velocity(phase, R_arr[i - 1])
        tau = 1.0 if phase in (HOVER_UP, HOVER_DOWN) else 2.0
        alpha = min(1.0, dt_sim / tau)
        v_arr[i] = v_physics + alpha * (v_target - v_physics)

        if phase == CRUISE:
            v_arr[i, 2] = 0.0
            R_arr[i - 1, 2] = z_cruise

        # ── Theory Step 3: v(t, θ, φ, γ, x, y, z) = v(t) + v_w(x, y, z) ──
        v_with_wind = v_arr[i] + v_wind

        # ── Theory Step 4: R(t+Δt) = R(t) + v(t, θ, φ, γ, x, y, z) × Δt ──
        R_arr[i] = R_arr[i - 1] + v_with_wind * dt_sim

        frac = np.hypot(R_arr[i, 0], R_arr[i, 1]) / max(np.hypot(x_dest, y_dest), 1.0)
        ground = z_origin + frac * (z_dest - z_origin)
        if R_arr[i, 2] < ground:
            R_arr[i, 2] = ground
            v_arr[i, 2] = max(v_arr[i, 2], 0.0)

        pw = phase_power[phase] * 1.25
        energy_kwh += pw * (dt_sim / 3600.0)
        phase_arr[i] = phase
        power_arr[i] = pw
        step = i

    n = step + 1
    t_arr = t_arr[:n]
    v_arr = v_arr[:n]
    R_arr = R_arr[:n]
    phase_arr = phase_arr[:n]
    power_arr = power_arr[:n]

    dist_flown_m = float(np.sum(np.linalg.norm(np.diff(R_arr, axis=0), axis=1)))
    return t_arr, v_arr, R_arr, phase_arr, power_arr, float(energy_kwh), dist_flown_m


def _color_arr(phase_arr):
    return [phase_colors.get(p, "gray") for p in phase_arr]


def _phase_legend(target, **kw):
    from matplotlib.patches import Patch
    h = [Patch(facecolor=phase_colors[p], label=phase_names[p]) for p in range(5)]
    d = dict(loc="lower center", ncol=5, fontsize=8, framealpha=0.9)
    d.update(kw)
    return target.legend(handles=h, **d)


def plot_2d(t_arr: np.ndarray, v_arr: np.ndarray, R_arr: np.ndarray,
            phase_arr: np.ndarray, power_arr: np.ndarray, battery_floor: float):
    colors = _color_arr(phase_arr)
    dist_along_km = np.hypot(R_arr[:, 0], R_arr[:, 1]) / 1000.0
    alt_ft = R_arr[:, 2] * 3.28084
    speed_kt = np.linalg.norm(v_arr, axis=1) * 1.94384
    t_min = t_arr / 60.0
    energy_cum = np.cumsum(power_arr * np.diff(t_arr, prepend=0) / 3600.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(dist_along_km, alt_ft, c=colors, s=2)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Altitude (ft)")

    ax = axes[0, 1]
    ax.scatter(t_min, speed_kt, c=colors, s=2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Speed (kt)")

    ax = axes[1, 0]
    ax.scatter(t_min, power_arr, c=colors, s=2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Power (kW)")

    ax = axes[1, 1]
    ax.plot(t_min, energy_cum, "b-", linewidth=1.2, label="Energy used")
    _reserve = (reserve_min / 60.0) * power_cruise * 1.25
    ax.axhline(y=battery_floor, linestyle="--", linewidth=1.0,
               color="r", label=f"Battery floor ({battery_floor:.0f} kWh)")
    ax.axhline(y=_reserve, linestyle=":", linewidth=1.0,
               color="orange", label=f"Reserve ({_reserve:.0f} kWh)")
    ax.axhline(y=battery_kwh, linestyle="-", linewidth=1.0, alpha=0.25,
               color="k", label=f"Capacity ({battery_kwh:.0f} kWh)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Energy (kWh)")
    ax.legend(fontsize=8)

    _phase_legend(fig, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("mission_result.png", dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_3d(t_arr: np.ndarray, v_arr: np.ndarray, R_arr: np.ndarray, phase_arr: np.ndarray):
    colors = _color_arr(phase_arr)
    lat0, lon0 = orig["lat"], orig["lon"]
    lat0_r = np.radians(lat0)
    lats = lat0 + np.degrees(R_arr[:, 1] / R_EARTH)
    lons = lon0 + np.degrees(R_arr[:, 0] / (R_EARTH * np.cos(lat0_r)))
    t_min = t_arr / 60.0

    fig = plt.figure(figsize=(16, 14))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    for pid in range(5):
        mask = phase_arr == pid
        if np.any(mask):
            idx = np.where(mask)[0]
            ax.plot(R_arr[idx, 0] / 1000, R_arr[idx, 1] / 1000, R_arr[idx, 2],
                    color=phase_colors[pid], linewidth=2.0, label=phase_names[pid])
    ax.scatter(*R_arr[0] / np.array([1000, 1000, 1]),
               color="green", s=80, marker="^", zorder=5)
    ax.scatter(R_arr[-1, 0] / 1000, R_arr[-1, 1] / 1000, R_arr[-1, 2],
               color="red", s=80, marker="v", zorder=5)
    ax.plot(R_arr[:, 0] / 1000, R_arr[:, 1] / 1000, np.zeros(len(R_arr)),
            "k--", alpha=0.2, linewidth=0.8)
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_zlabel("Altitude (m)")
    ax.legend(fontsize=7, loc="upper left")

    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(lons, lats, c=colors, s=1)
    for nid, n in nodes.items():
        ax.plot(n["lon"], n["lat"], "ks", markersize=5)
        ax.annotate(nid, (n["lon"], n["lat"]), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t_min, v_arr[:, 0] * 1.94384, label="v_east", alpha=0.8)
    ax.plot(t_min, v_arr[:, 1] * 1.94384, label="v_north", alpha=0.8)
    ax.plot(t_min, v_arr[:, 2] * 1.94384, label="v_up", alpha=0.8)
    ax.plot(t_min, np.linalg.norm(v_arr, axis=1) * 1.94384,
            "k-", linewidth=1.5, label="|v| total")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Velocity (kt)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(t_min, R_arr[:, 2] * 3.28084, c=colors, s=2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Altitude (ft)")
    ax.grid(True, alpha=0.3)

    _phase_legend(fig, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("mission_3d.png", dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ── appended: map visualization ──────────────────────────────────────────────

def plot_map(R_arr: np.ndarray, phase_arr: np.ndarray,
             dist_flown: float, t_arr: np.ndarray, energy_kwh: float):
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    lat0, lon0 = orig["lat"], orig["lon"]
    lat0_r = np.radians(lat0)
    lats = lat0 + np.degrees(R_arr[:, 1] / R_EARTH)
    lons = lon0 + np.degrees(R_arr[:, 0] / (R_EARTH * np.cos(lat0_r)))

    coast = np.array([
        [-122.50, 37.00], [-122.45, 37.10], [-122.52, 37.20], [-122.52, 37.40],
        [-122.50, 37.50], [-122.52, 37.60], [-122.52, 37.70], [-122.48, 37.78],
        [-122.50, 37.82], [-122.50, 37.90], [-122.44, 37.95], [-122.42, 38.00],
        [-122.38, 38.05], [-122.36, 38.10], [-122.38, 38.15], [-122.40, 38.20],
        [-122.42, 38.30], [-122.45, 38.40], [-122.50, 38.50], [-122.52, 38.60],
        [-122.55, 38.70], [-122.50, 38.80],
    ])
    sf_bay = np.array([
        [-122.48, 37.78], [-122.42, 37.80], [-122.38, 37.82], [-122.35, 37.85],
        [-122.30, 37.88], [-122.22, 37.90], [-122.15, 37.92], [-122.10, 37.95],
        [-122.08, 38.00], [-122.05, 38.03], [-122.05, 38.06], [-122.10, 38.08],
        [-122.15, 38.10], [-122.20, 38.12], [-122.25, 38.10], [-122.28, 38.06],
        [-122.30, 38.02], [-122.28, 37.98], [-122.22, 37.95], [-122.20, 37.90],
        [-122.18, 37.85], [-122.15, 37.80], [-122.12, 37.75], [-122.10, 37.70],
        [-122.08, 37.65], [-122.05, 37.60], [-122.03, 37.55], [-122.05, 37.50],
        [-122.10, 37.48], [-122.15, 37.45], [-122.20, 37.43],
    ])
    south_bay = np.array([
        [-122.20, 37.43], [-122.15, 37.38], [-122.10, 37.35], [-122.05, 37.33],
        [-122.00, 37.32], [-121.95, 37.33], [-121.93, 37.35], [-121.93, 37.40],
        [-121.95, 37.45], [-121.97, 37.50], [-122.00, 37.55], [-122.03, 37.55],
    ])
    delta = np.array([
        [-121.85, 38.05], [-121.80, 38.10], [-121.75, 38.15], [-121.70, 38.20],
        [-121.65, 38.25], [-121.60, 38.30], [-121.55, 38.35], [-121.60, 38.40],
        [-121.65, 38.45], [-121.70, 38.50], [-121.75, 38.55],
    ])
    cities = {
        "San Francisco": (-122.42, 37.77),
        "Oakland":       (-122.27, 37.80),
        "San Jose":      (-121.89, 37.34),
        "Sacramento":    (-121.49, 38.58),
        "Stockton":      (-121.29, 37.95),
        "Santa Cruz":    (-122.03, 36.97),
    }

    fig, ax = plt.subplots(figsize=(12, 14))

    ax.fill([-123, -120, -120, -123], [36.5, 36.5, 39, 39], color="#f0ead6", zorder=0)
    ocean_x = list(coast[:, 0]) + [-123.0, -123.0]
    ocean_y = list(coast[:, 1]) + [coast[-1, 1], coast[0, 1]]
    ax.fill(ocean_x, ocean_y, color="#d4e6f1", zorder=1)
    ax.fill(sf_bay[:, 0], sf_bay[:, 1], color="#d4e6f1", zorder=2)
    ax.fill(south_bay[:, 0], south_bay[:, 1], color="#d4e6f1", zorder=2)
    ax.plot(coast[:, 0], coast[:, 1], "k-", linewidth=0.8, zorder=3)
    ax.plot(sf_bay[:, 0], sf_bay[:, 1], "k-", linewidth=0.6, alpha=0.5, zorder=3)
    ax.plot(south_bay[:, 0], south_bay[:, 1], "k-", linewidth=0.6, alpha=0.5, zorder=3)
    ax.plot(delta[:, 0], delta[:, 1], color="#85c1e9", linewidth=1.5, alpha=0.6, zorder=3)

    for city, (cx, cy) in cities.items():
        ax.plot(cx, cy, "o", color="gray", markersize=4, zorder=5)
        ax.annotate(city, (cx, cy), fontsize=7, color="gray",
                    xytext=(6, 3), textcoords="offset points", zorder=5)

    for pid in [HOVER_UP, CLIMB, CRUISE, DESCENT, HOVER_DOWN]:
        mask = phase_arr == pid
        if np.any(mask):
            idx = np.where(mask)[0]
            ax.plot(lons[idx], lats[idx], color=phase_colors[pid], linewidth=3.5,
                    zorder=7, solid_capstyle="round",
                    path_effects=[pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])

    for nid, n in nodes.items():
        ax.plot(n["lon"], n["lat"], "o", color="white", markersize=14,
                markeredgecolor="black", markeredgewidth=1.5, zorder=9)
        ax.plot(n["lon"], n["lat"], "H", color="#2c3e50", markersize=7, zorder=10)
        ax.annotate(n["name"], (n["lon"], n["lat"]), fontsize=8, fontweight="bold",
                    xytext=(14, 6), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#2c3e50", alpha=0.9), zorder=11)

    ax.plot(orig["lon"], orig["lat"], "^", color="#27ae60", markersize=18,
            markeredgecolor="black", markeredgewidth=1.2, zorder=12)
    ax.plot(dest["lon"], dest["lat"], "v", color="#e74c3c", markersize=18,
            markeredgecolor="black", markeredgewidth=1.2, zorder=12)

    inset = fig.add_axes([0.12, 0.08, 0.35, 0.12])
    dist_along_km = np.hypot(R_arr[:, 0], R_arr[:, 1]) / 1000.0
    inset.scatter(dist_along_km, R_arr[:, 2] * 3.28084, c=_color_arr(phase_arr), s=1)
    inset.set_xlabel("Distance (km)", fontsize=7)
    inset.set_ylabel("Alt (ft)", fontsize=7)
    inset.set_title("Vertical Profile", fontsize=8, fontweight="bold")
    inset.tick_params(labelsize=6)
    inset.set_facecolor("#f9f9f9")
    inset.grid(True, alpha=0.2)

    handles = [Line2D([0], [0], color=phase_colors[p], linewidth=3, label=phase_names[p])
               for p in [HOVER_UP, CLIMB, CRUISE, DESCENT, HOVER_DOWN]]
    handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="#27ae60",
                          markersize=10, markeredgecolor="black", label="Origin"))
    handles.append(Line2D([0], [0], marker="v", color="w", markerfacecolor="#e74c3c",
                          markersize=10, markeredgecolor="black", label="Destination"))
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9,
              title="Flight Phase", title_fontsize=10)

    ax.set_xlim(-122.6, -120.2)
    ax.set_ylim(36.8, 38.8)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(
        f"Airlink eVolve  \u2014  {orig['name']} \u2192 {dest['name']}\n"
        f"Joby S4  |  {dist_flown/1852:.1f} NM  |  {t_arr[-1]/60:.1f} min  |  "
        f"{energy_kwh:.1f} kWh",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)

    plt.savefig("mission_map.png", dpi=180, bbox_inches="tight", facecolor="white")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────

SHOW_PLOTS = True

t_arr, v_arr, R_arr, phase_arr, power_arr, energy_kwh, dist_flown = simulate()
reserve_kwh = (reserve_min / 60.0) * power_cruise * 1.25
battery_floor = energy_kwh + reserve_kwh
plot_2d(t_arr, v_arr, R_arr, phase_arr, power_arr, battery_floor)
plot_3d(t_arr, v_arr, R_arr, phase_arr)
plot_map(R_arr, phase_arr, dist_flown, t_arr, energy_kwh)