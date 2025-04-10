import numpy as np
from utils import *
import utils
import plant
import control as co
from control import matlab
from scipy import signal
import pickle
import sys
import print_ASCII as pa
import matplotlib.pyplot as plt
# prints ASCII art of the system
pa.print_system()
print("Running Simple DOB for VCM Rotational Vibration")

#sampling time and Multi-rate number
Ts = plant.Ts
Mr_f = plant.Mr_f

# Controllers
Sys_Cd_pzt = get_Sys_Cd_pzt()
Sys_Cd_vcm = get_Sys_Cd_vcm()
Sys_Fm_vcm = get_Sys_Fm_vcm()
Sys_Fm_pzt = get_Sys_Fm_pzt()

# Function to create controlled objects (Discrete-time system)
def create_controlled_objects(Sys_Pc_vcm, Sys_Pc_pzt):
    # Convert the continuous-time VCM plant model to discrete-time using Zero-Order Hold (ZOH) method
    # with a sampling time of Ts/Mr_f, where Ts is the overall sampling time and Mr_f is the multi-rate factor
    # return a Ts sampling time VCM PZT model

    Sys_Pdm0_vcm = matlab.c2d(Sys_Pc_vcm, Ts/Mr_f, 'zoh')
    Sys_Pdm_vcm = Sys_Pdm0_vcm * Sys_Fm_vcm
    #Sys_Pd_vcm = Sys_Pdm0_vcm * Sys_Fm_vcm
    Sys_Pd_vcm = dts_resampling(Sys_Pdm_vcm, Mr_f)

    Sys_Pdm0_pzt = matlab.c2d(Sys_Pc_pzt, Ts/Mr_f, 'zoh')
    Sys_Pdm_pzt = Sys_Pdm0_pzt * Sys_Fm_pzt
    #Sys_Pd_pzt = Sys_Pdm0_vcm * Sys_Fm_vcm
    Sys_Pd_pzt = dts_resampling(Sys_Pdm_pzt, Mr_f)

    return Sys_Pd_vcm, Sys_Pd_pzt

# Create controlled objects for each case 
Sys_Pd_vcm_all = []
Sys_Pd_pzt_all = []
for i in range(1, 10):
    Sys_Pc_vcm = getattr(plant, f"Sys_Pc_vcm_c{i}")
    Sys_Pc_pzt = getattr(plant, f"Sys_Pc_pzt_c{i}")
    Sys_Pd_vcm, Sys_Pd_pzt = create_controlled_objects(Sys_Pc_vcm, Sys_Pc_pzt)
    Sys_Pd_vcm_all.append(Sys_Pd_vcm)
    Sys_Pd_pzt_all.append(Sys_Pd_pzt)

# Load simulation results
try:
    sim_results = []
    for i in range(1, 10):
        sim_result = pickle.load(open(utils.get_sim_path(f"res{i}.pkl"), "rb"))
        sim_results.append(sim_result)
except Exception as e:
    print("ERROR: Simulation files not found, have you run function_simulation.py first? exiting ...")
    sys.exit()

# Plot rotational vibration (df)
plt.figure(30)
plt.plot(sim_results[0]["time"][1:420*20] * 1e3, sim_results[0]["df"][1:420*20] * 1e9)
plt.title('Rotational Vibration ($d_f$)')
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [nm]')
plt.grid(True)
plt.savefig(utils.get_plot_path("figure30_rotational_vibration.png"))

# Create nominal plant model (using Case 2 - RT)
print("Using RT (Case 2) as nominal model")
P_nom = Sys_Pd_vcm_all[1]

# Time vector and input
time = np.arange(0, Ts*420, Ts)
u = np.ones_like(time)  # Step input

# Get disturbance signal
df_data = sim_results[0]["df"][0:420*20]
df = df_data[::20]  # Downsample to match time vector

# System response with disturbance
u_with_dis = u + df
y = co.forced_response(P_nom, T=time, U=u_with_dis)[1]
y_step = co.forced_response(P_nom, T=time, U=u)[1]  # Reference response

# Design simple DOB for rotational vibration
print("Designing simple DOB for rotational vibration...")

# 1. Get nominal output and disturbance effect
y_nom = co.forced_response(P_nom, T=time, U=u)[1]
dist_effect = y - y_nom

# 2. Design and apply low-pass filter
nyquist = 1/(2*Ts)
cutoff_freq = 10.0  # 10Hz cutoff
normalized_cutoff = cutoff_freq / nyquist
b_low, a_low = signal.butter(2, normalized_cutoff, 'low')
filtered_dist = signal.filtfilt(b_low, a_low, dist_effect)

# 3. Apply moving average smoothing
window_size = 20
smoother = np.ones(window_size) / window_size
smoothed_dist = np.convolve(filtered_dist, smoother, mode='same')

# 4. Apply DOB compensation
compensation_scale = 0.01
u_comp = u - smoothed_dist * compensation_scale

# 5. Add small delay for causality
delay_samples = 2
u_comp = np.roll(u_comp, delay_samples)
u_comp[:delay_samples] = u[:delay_samples]

# Get compensated response
y_comp = co.forced_response(P_nom, T=time, U=u_comp + df)[1]

# Plot results
# 1. Disturbance estimation
plt.figure(31)
plt.plot(time * 1e3, df * 1e9, label="Actual Rotational Vibration", alpha=0.7)
plt.plot(time * 1e3, smoothed_dist * 1e9, label="DOB Estimate", linestyle='-')
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [nm]')
plt.legend()
plt.grid(True)
plt.title('DOB Estimation of Rotational Vibration')
plt.savefig(utils.get_plot_path("figure31_dob_estimation.png"))

# 2. System response comparison
plt.figure(32)
plt.plot(time * 1e3, y_step, label="Reference (No Disturbance)")
plt.plot(time * 1e3, y, label="With Rotational Vibration")
plt.plot(time * 1e3, y_comp, label="With DOB Compensation")
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.title('System Response Comparison')
plt.savefig(utils.get_plot_path("figure32_response_comparison.png"))

# 3. Error comparison
plt.figure(33)
error_without_comp = y - y_step
error_with_comp = y_comp - y_step

plt.plot(time * 1e3, error_without_comp * 1e9, 'r-', label='Without DOB')
plt.plot(time * 1e3, error_with_comp * 1e9, 'b-', label='With DOB')
plt.xlabel('Time [ms]')
plt.ylabel('Error [nm]')
plt.legend()
plt.grid(True)
plt.title('Error Comparison')
plt.savefig(utils.get_plot_path("figure33_error_comparison.png"))

# Calculate performance metrics
error_without_comp_rms = np.sqrt(np.mean(error_without_comp**2))
error_with_comp_rms = np.sqrt(np.mean(error_with_comp**2))
improvement_rms = (1 - error_with_comp_rms / error_without_comp_rms) * 100

max_error_without = np.max(np.abs(error_without_comp))
max_error_with = np.max(np.abs(error_with_comp))
improvement_max = (1 - max_error_with / max_error_without) * 100

print("\nPerformance Results:")
print(f"RMS Error without DOB: {error_without_comp_rms:.6e}")
print(f"RMS Error with DOB: {error_with_comp_rms:.6e}")
print(f"RMS Error Improvement: {improvement_rms:.2f}%")
print(f"\nMax Error without DOB: {max_error_without:.6e}")
print(f"Max Error with DOB: {max_error_with:.6e}")
print(f"Max Error Improvement: {improvement_max:.2f}%")

plt.show()

