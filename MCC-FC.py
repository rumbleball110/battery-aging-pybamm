# %%
import pybamm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Set logging 
pybamm.set_logging_level("INFO")

# DFN Model with different aging mechanisms
model = pybamm.lithium_ion.DFN(
    options={
        "SEI": "solvent-diffusion limited",  
        "lithium plating": "partially reversible",  
        "particle mechanics": "swelling and cracking",  
        "thermal": "lumped",  
        "loss of active material": "stress-driven",
    },
    name="DFN with SEI, Plating, and Cracking"
)

# Load OKane2022 parameter set for realistic aging parameters
param = pybamm.ParameterValues("OKane2022")


param.update({
    # SEI growth
    "SEI kinetic rate constant [m.s-1]": 1e-13,        
    "SEI resistivity [Ohm.m]": 1e5,                  
    "Initial SEI thickness [m]": 1e-9,                 

    # Lithium plating
    "Lithium plating kinetic rate constant [m.s-1]": 5e-10, 

    # LAM 
    "Negative electrode LAM constant proportional term [s-1]": 1e-4 / 3600, 
    "Positive electrode LAM constant proportional term [s-1]": 1e-4 / 3600,

    "Reference temperature [K]": 298.15 
})


# Define experiment - Multi Stage Constant Current Fast Charging (MCC-FC)
C = param["Nominal cell capacity [A.h]"]
soc_stage = 0.25 
currents = [3.0, 2.0, 1.0, 0.5] 
durations = [soc_stage * C / I * 60 for I in currents]

experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 3.0V",              
        "Rest for 30 minutes",
        f"Charge at {currents[0]}C for {durations[0]:.1f} minutes or until 4.05 V",
        f"Charge at {currents[1]}C for {durations[1]:.1f} minutes or until 4.05 V",
        f"Charge at {currents[2]}C for {durations[2]:.1f} minutes or until 4.05 V",
        f"Charge at {currents[3]}C for {durations[3]:.1f} minutes or until 4.05 V",
        "Hold at 4.1 V until 0.05C",                
        "Rest for 30 minutes"
    ] * 100,  
    period="1 minute"
)

# Set up numerical solver 
solver = pybamm.CasadiSolver(mode="safe", atol=1e-2,  rtol=1e-2)

# Create and solve simulation
sim = pybamm.Simulation(
    model=model,
    parameter_values=param,
    experiment=experiment,
    solver=solver
)

# Solve the model
solution = sim.solve()

# %%
# Extract data
time = solution["Time [min]"].entries
discharge_capacity = solution["Discharge capacity [A.h]"].entries
current = solution["Current [A]"].entries
voltage = solution["Terminal voltage [V]"].entries
lithium_loss = solution["Loss of lithium to negative SEI [mol]"].entries
sei_thickness = solution["X-averaged negative SEI thickness [m]"].entries * 1e9 
plated_lithium = solution["Loss of capacity to negative lithium plating [A.h]"].entries
lam_n = solution["X-averaged negative electrode active material volume fraction"].entries
lam_n0 = lam_n[0]
lam_n = 100 * (1 - lam_n / lam_n0)
stress = solution["X-averaged negative particle surface tangential stress [Pa]"].entries
temperature = solution["Volume-averaged cell temperature [K]"].entries

# Save results to CSV
results_df = pd.DataFrame({
    "Time [min]": time,
    "Voltage [V]": voltage,
    "Discharge Capacity [A.h]": discharge_capacity,
    "Current [A]": current,
    "Lithium Loss [mol m-3]": lithium_loss,
    "Plated Lithium Capacity [A.h]": plated_lithium,
    "LAM Negative [%]": lam_n,
    "Electrode Stress [MPa]": stress / 1e6,
    "Temperature [K]": temperature,
    "SEI Thickness [nm]": sei_thickness
})
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df.to_csv(f"aging_results_{timestamp}.csv", index=False)

# Save cycle-wise degradation summary (capacity, plating, LAM, fade)
cycle_summaries = []

# Calculate cycle-wise capacity and normalize - to store in a dataframe
initial_capacity = None  # Will be set from first cycle

for i, cyc in enumerate(solution.cycles):
    cap = cyc["Discharge capacity [A.h]"].entries[-1]
    plat = cyc["Loss of capacity to negative lithium plating [A.h]"].entries[-1]
    lam = 100 * (1 - cyc["X-averaged negative electrode active material volume fraction"].entries[-1] / lam_n0)
    
    if initial_capacity is None:
        initial_capacity = cap  # Set the reference capacity

    rel_capacity = 100 * (cap / initial_capacity)
    fade = 100 * (1 - cap / initial_capacity)

    cycle_summaries.append({
        "Cycle": i + 1,
        "Capacity [A.h]": cap,
        "Relative Capacity [%]": rel_capacity,
        "Capacity Fade [%]": fade,
        "Plated [A.h]": plat,
        "LAM [%]": lam
    })

# Create DataFrame and save
df_cycles = pd.DataFrame(cycle_summaries)
df_cycles.to_csv(f"cyclewise_degradation_{timestamp}.csv", index=False)

# Make sure LAM [%] is cleaned 
import ast
def parse_last_lam(val):
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val)
        except Exception:
            return np.nan
    if isinstance(val, (list, np.ndarray)) and len(val) > 0:
        return val[-1]
    return val

df_cycles["LAM [%]"] = df_cycles["LAM [%]"].apply(parse_last_lam)

# Filter valid data
df_clean = df_cycles.dropna(subset=["Capacity Fade [%]", "Plated [A.h]", "LAM [%]"])

# Filter only 1st, 9th, 17th... cycles - Each step in this experiment is taken as cycle and there are 8 steps in a cycle
filtered_df = df_clean[df_clean["Cycle"] % 8 == 1].copy()

# Extract filtered X and Y values
cycles = filtered_df["Cycle"]/8
fade = filtered_df["Capacity Fade [%]"]
plated = filtered_df["Plated [A.h]"]
lam_ne = filtered_df["LAM [%]"]

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot 1: Capacity Fade
axs[0].plot(cycles, fade, marker='o', color='tab:blue')
axs[0].set_ylabel("Capacity Fade [%]")
axs[0].set_title("Capacity Fade vs Cycle")
axs[0].grid(True)

# Plot 2: Plated Lithium
axs[1].plot(cycles, plated, marker='x', color='tab:orange')
axs[1].set_ylabel("Plated Lithium [A.h]")
axs[1].set_title("Plated Lithium vs Cycle")
axs[1].grid(True)

# Plot 3: LAM
axs[2].plot(cycles, lam_ne, marker='s', color='tab:green')
axs[2].set_xlabel("Cycle Number")
axs[2].set_ylabel("LAM Negative [%]")
axs[2].set_title("LAM (Negative Electrode) vs Cycle")
axs[2].grid(True)

# Final layout
plt.tight_layout()
plt.show()

# Plot SEI thickness over time
plt.figure(figsize=(10, 5))
plt.plot(time / 60, sei_thickness, color='tab:blue', linewidth=2)
plt.xlabel("Time [h]")
plt.ylabel("SEI Thickness [nm]")
plt.title("SEI Thickness vs Time")
plt.grid(True)
plt.tight_layout()
plt.show()


# Create comprehensive visualizations
fig = plt.figure(figsize=(12, 10))

# Plot 1: Voltage, Capacity, and Current
ax1 = fig.add_subplot(311)
ax1.plot(time / 60, discharge_capacity, marker='o', color='tab:blue', label='Discharge Capacity', markersize=3)
ax1.set_xlabel("Time [h]")
ax1.set_ylabel("Capacity [A.h]", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(time / 60, current, marker='x', color='tab:red', label='Current', markersize=3)
ax2.set_ylabel("Current [A]", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.plot(time / 60, voltage, marker='^', color='tab:green', label='Voltage', markersize=3)
ax3.set_ylabel("Voltage [V]", color='tab:green')
ax3.tick_params(axis='y', labelcolor='tab:green')
ax1.set_title("Voltage, Capacity, and Current vs Time")

# Plot 2: Plated Lithium and LAM
ax4 = fig.add_subplot(312)
ax4.plot(time / 60, plated_lithium, marker='o', color='tab:brown', label='Plated Lithium Capacity', markersize=3)
ax4.set_xlabel("Time [h]")
ax4.set_ylabel("Plated Capacity [A.h]", color='tab:brown')
ax4.tick_params(axis='y', labelcolor='tab:brown')
ax4.grid(True)

ax5 = ax4.twinx()
ax5.plot(time / 60, lam_n, marker='x', color='tab:gray', label='LAM Negative Electrode', markersize=3)
ax5.set_ylabel("LAM [%]", color='tab:gray')
ax5.tick_params(axis='y', labelcolor='tab:gray')
ax4.set_title("Plated Lithium and LAM vs Time")



# Plot 3: Stress and Temperature
ax6 = fig.add_subplot(313)
ax6.plot(time / 60, -stress / 1e6, marker='o', color='tab:cyan', label='Electrode Stress', markersize=3)
ax6.set_xlabel("Time [h]")
ax6.set_ylabel("Negative Particle Surface Tangential Stress [MPa]", color='tab:cyan')
ax6.tick_params(axis='y', labelcolor='tab:cyan')
ax6.grid(True)

ax7 = ax6.twinx()
ax7.plot(time / 60, temperature - 273.15, marker='x', color='tab:pink', label='Temperature', markersize=3)
ax7.set_ylabel("Temperature [°C]", color='tab:pink')
ax7.tick_params(axis='y', labelcolor='tab:pink')
ax6.set_title("Electrode Stress and Temperature vs Time")

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax3.legend(loc="center right")

ax4.legend(loc="upper left")
ax5.legend(loc="upper right")

ax6.legend(loc="upper left")
ax7.legend(loc="upper right")


fig.tight_layout()
plt.show()

# Calculate summary statistics using every 8th cycle starting from cycle 1
selected_capacities = []
selected_plated = []
selected_cycles = []
selected_sei = []
stress_stats = []
for i, cyc in enumerate(solution.cycles):
    if i % 8 == 0:  # 0-indexed → 1st, 9th, 17th, etc.
        cap = cyc["Discharge capacity [A.h]"].entries[-1]
        plat = cyc["Loss of capacity to negative lithium plating [A.h]"].entries[-1]
        sei = cyc["X-averaged negative SEI thickness [m]"].entries[-1] * 1e9  # in nm
        selected_capacities.append(cap)
        selected_plated.append(plat)
        selected_sei.append(sei)
        selected_cycles.append((i + 1)/8)
        cyc_stress = cyc["X-averaged negative particle surface tangential stress [Pa]"].entries
        stress_range = np.max(cyc_stress) - np.min(cyc_stress)
        stress_mean = np.mean(cyc_stress)
        stress_stats.append({
            "Cycle": i + 1,
            "Stress Mean [MPa]": stress_mean / 1e6,
            "Stress Range [MPa]": stress_range / 1e6
        })
df_stress = pd.DataFrame(stress_stats)

# Ensure valid selection of cycles
if selected_capacities:
    initial_capacity = selected_capacities[0]
    final_capacity = selected_capacities[-1]
    capacity_fade = 100 * (initial_capacity - final_capacity) / initial_capacity

    initial_plated = selected_plated[0]
    final_plated_capacity = selected_plated[-1]
else:
    print("Warning: No valid cycles selected for fade computation.")
    initial_capacity = final_capacity = capacity_fade = np.nan
    initial_plated = final_plated_capacity = np.nan

final_lam_n = lam_n[-1]
max_stress = np.max(stress) / 1e6
max_temperature = np.max(temperature) - 273.15
final_sei_thickness = selected_sei[-1] if selected_sei else np.nan


# Print summary
print("\n=== Summary (Based on 1st, 9th, 17th... Cycles) ===")
print(f"Initial Capacity: {initial_capacity:.3f} A.h")
print(f"Final Capacity:   {final_capacity:.3f} A.h")
print(f"Capacity Fade:    {capacity_fade:.2f}%")
print(f"Initial Plated Lithium: {initial_plated:.6f} A.h")
print(f"Final SEI Thickness:              {final_sei_thickness:.2f} nm")
print(f"Final Plated Lithium:   {final_plated_capacity:.6f} A.h")
print(f"Final LAM in Negative Electrode: {final_lam_n:.6f} %")
print(f"Maximum Electrode Stress:        {max_stress:.2f} MPa")
print(f"Maximum Temperature:             {max_temperature:.2f} °C")

sei_growth_per_cycle = np.diff(selected_sei)
plt.figure(figsize=(10, 4))
plt.plot(selected_cycles[1:], sei_growth_per_cycle, marker='o')
plt.xlabel("Cycle")
plt.ylabel("SEI Growth [nm/cycle]")
plt.title("SEI Growth Rate Over Cycles")
plt.grid(True)
plt.tight_layout()
plt.show()
