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
print("Running DOB for VCM v0.1.1")

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


# Disturbance Observer (DOB) of P_VCM

# Simulation
# Load simulation results
try:
    sim_results = []
    for i in range(1, 10):
        sim_result = pickle.load(open(utils.get_sim_path(f"res{i}.pkl"), "rb"))
        sim_results.append(sim_result)
except Exception as e:
    print("ERROR: Simulation files not found, have you run function_simulation.py first? exiting ...")
    sys.exit()

# Figure 20: df - Rotational Vibration
plt.figure(20)
# translate to ms and nm
plt.plot(sim_results[0]["time"][1:420*20] * 1e3, sim_results[0]["df"][1:420*20] * 1e9)
plt.title('$d_f$ - Rotational Vibration')
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [nm]')
plt.grid(True)
plt.xlim([0, Ts*420*1e3])
plt.savefig(utils.get_plot_path("figure20_df_rotational_vibration.png"))

length_of_slice = len(sim_results[0]["time"][1:420*20])
print("Length of sim_results[0]['time'][1:420*20]:", length_of_slice)

print("Ts*420*1e3:", Ts*420*1e3)

# Create a nominal plant model for DOB
print("Using RT (Case 2) as nominal model")
P_nom = Sys_Pd_vcm_all[1]  # Using RT (Case 2) as nominal model

# Simulate the system with disturbance
time = np.arange(0, Ts*420, Ts)  # Time vector
print("Length of time vector:", len(time))

# Step input
u = np.ones_like(time)  
print("Length of input vector:", len(u))

# Get disturbance signal (with proper resampling)
df_data = sim_results[0]["df"][0:420*20]  # Original disturbance data
df = df_data[::20]  # Downsample to match our time vector
print("Length of disturbance vector:", len(df))

# Input with disturbance
u_with_dis = u + df

# System response to input with disturbance
y = co.forced_response(P_nom, T=time, U=u_with_dis)[1]
print("Length of output vector:", len(y))

# Step response without disturbance for comparison
y_step = co.forced_response(P_nom, T=time, U=u)[1]

# Plot step responses with and without disturbance
plt.figure(23)
plt.plot(time * 1e3, y_step, label="Step response (no disturbance)")
plt.plot(time * 1e3, y, label="Response with rotational vibration", linestyle='--')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.title('System Response With and Without Rotational Vibration')
plt.savefig(utils.get_plot_path("figure23_response_with_rotational_vibration.png"))

# Plot error due to disturbance (magnify the difference)
plt.figure(25)
error_due_to_disturbance = y - y_step
plt.plot(time * 1e3, error_due_to_disturbance * 1e9, 'r-')
plt.xlabel('Time (ms)')
plt.ylabel('Error (nm)')
plt.title('Error Due to Rotational Vibration (Magnified)')
plt.grid(True)
plt.savefig(utils.get_plot_path("figure25_error_due_to_rotational_vibration.png"))

# ----- Part 1: Direct feedforward compensation with known disturbance (benchmark) -----
print("Direct feedforward compensation experiment...")
print("Using the actual disturbance for compensation (not DOB)")

# Known disturbance (as a benchmark/reference)
known_disturbance = df

# Compensation scale for direct feedforward
compensation_scale = 0.05
u_comp_direct = u - known_disturbance * compensation_scale

# Apply a small delay for realism
delay_samples = 2
u_comp_direct = np.roll(u_comp_direct, delay_samples)
u_comp_direct[:delay_samples] = u[:delay_samples]  # No compensation at the start

# System response with direct compensation
y_comp_direct = co.forced_response(P_nom, T=time, U=u_comp_direct + df)[1]

# ----- Part 2: VCM Rotational Vibration DOB Design -----
print("Computing VCM Rotational Vibration DOB...")

# Calculate nominal output
y_nom = co.forced_response(P_nom, T=time, U=u)[1]

# Calculate disturbance effect in output
dist_effect_raw = y - y_nom

# Analyze the frequency content of rotational vibration (df)
from scipy import fftpack
# Get the frequency spectrum of the disturbance
n = len(df)
df_fft = fftpack.fft(df)
df_freq = fftpack.fftfreq(n, Ts)
df_magnitude = np.abs(df_fft)
# Only look at positive frequencies up to Nyquist frequency
pos_mask = np.where(df_freq > 0)
pos_freqs = df_freq[pos_mask]
pos_magnitude = df_magnitude[pos_mask]

# Plot the frequency spectrum of rotational vibration
plt.figure(27)
plt.plot(pos_freqs, pos_magnitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Rotational Vibration')
plt.grid(True)
plt.savefig(utils.get_plot_path("figure27_df_frequency_spectrum.png"))

# Find dominant frequencies
dominant_idx = np.argsort(pos_magnitude)[-5:]  # Get indices of 5 largest peaks
dominant_freqs = pos_freqs[dominant_idx]
print("Dominant frequencies in rotational vibration:", dominant_freqs)

# ---- SIMPLIFIED CONSERVATIVE DOB APPROACH ----
print("Using simplified DOB design for rotational vibration")

# Step 1: Apply very low-pass filter to focus only on low-frequency components
nyquist = 1/(2*Ts)
cutoff_freq = 10.0  # 10Hz cutoff for rotational vibration
normalized_cutoff = cutoff_freq / nyquist
b_low, a_low = signal.butter(2, normalized_cutoff, 'low')
low_freq_dist = signal.filtfilt(b_low, a_low, dist_effect_raw)

# Step 2: Apply moving average filter for smoothing
window_size = 20  # Window for smoothing
smoother = np.ones(window_size) / window_size
smooth_dist_effect = np.convolve(low_freq_dist, smoother, mode='same')

# Step 3: Use appropriate compensation scale
dob_scale = 0.01
print(f"Using compensation scale: {dob_scale}")

# Apply DOB compensation
u_comp_dob = u - smooth_dist_effect * dob_scale

# Apply small delay for causality
delay_samples = 2
u_comp_dob = np.roll(u_comp_dob, delay_samples)
u_comp_dob[:delay_samples] = u[:delay_samples]

# System response with DOB compensation
y_comp_dob = co.forced_response(P_nom, T=time, U=u_comp_dob + df)[1]

# Plot the disturbance estimates
plt.figure(19)
plt.plot(time * 1e3, df * 1e9, label="Actual Rotational Vibration", alpha=0.7)
plt.plot(time * 1e3, dist_effect_raw * 1e9, label="Raw DOB Estimate", linestyle=':', alpha=0.3)
plt.plot(time * 1e3, smooth_dist_effect * 1e9, label="Smoothed DOB Output", linestyle='-')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (nm)')
plt.legend()
plt.grid(True)
plt.title('VCM Rotational Vibration DOB')
plt.savefig(utils.get_plot_path("figure19_vcm_rotational_vibration_dob.png"))

# Plot response comparison
plt.figure(24)
plt.plot(time * 1e3, y_step, label="Ideal (No Disturbance)")
plt.plot(time * 1e3, y, label="With Rotational Vibration", linestyle='--')
plt.plot(time * 1e3, y_comp_direct, label="Direct Compensation", linestyle='-.')
plt.plot(time * 1e3, y_comp_dob, label="DOB Compensation", linestyle=':')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.title('Response Comparison')
plt.savefig(utils.get_plot_path("figure24_rotational_vibration_comparison.png"))

# Plot error comparison
plt.figure(26)
error_without_comp = y - y_step
error_with_direct_comp = y_comp_direct - y_step
error_with_dob_comp = y_comp_dob - y_step

plt.plot(time * 1e3, error_without_comp * 1e9, 'r-', label='No Compensation')
plt.plot(time * 1e3, error_with_direct_comp * 1e9, 'g-', label='Direct Compensation')
plt.plot(time * 1e3, error_with_dob_comp * 1e9, 'b-', label='DOB Compensation')
plt.xlabel('Time (ms)')
plt.ylabel('Error (nm)')
plt.legend()
plt.grid(True)
plt.title('Error Comparison (Magnified)')
plt.savefig(utils.get_plot_path("figure26_error_comparison_rotational.png"))

# Calculate performance metrics
try:
    # Calculate RMS error for each method
    error_without_comp = np.sqrt(np.mean((y - y_step)**2))
    error_with_direct_comp = np.sqrt(np.mean((y_comp_direct - y_step)**2))
    error_with_dob_comp = np.sqrt(np.mean((y_comp_dob - y_step)**2))
    
    # Improvement percentages
    direct_improvement = (1 - error_with_direct_comp / error_without_comp) * 100
    dob_improvement = (1 - error_with_dob_comp / error_without_comp) * 100
    
    print("\nPerformance Comparison (RMS Error):")
    print(f"Direct feedforward: {direct_improvement:.2f}% improvement")
    print(f"VCM Rotational Vibration DOB: {dob_improvement:.2f}% improvement")
    
    print("\nRMS Error values:")
    print(f"Without compensation: {error_without_comp:.6e}")
    print(f"Direct feedforward: {error_with_direct_comp:.6e}")
    print(f"VCM Rotational Vibration DOB: {error_with_dob_comp:.6e}")
    
    # Maximum error metrics
    max_error_without = np.max(np.abs(y - y_step))
    max_error_direct = np.max(np.abs(y_comp_direct - y_step))
    max_error_dob = np.max(np.abs(y_comp_dob - y_step))
    
    max_direct_improvement = (1 - max_error_direct / max_error_without) * 100
    max_dob_improvement = (1 - max_error_dob / max_error_without) * 100
    
    print("\nPerformance Comparison (Maximum Error):")
    print(f"Direct feedforward: {max_direct_improvement:.2f}% improvement")
    print(f"VCM Rotational Vibration DOB: {max_dob_improvement:.2f}% improvement")
    
    print("\nMaximum Error values:")
    print(f"Without compensation: {max_error_without:.6e}")
    print(f"Direct feedforward: {max_error_direct:.6e}")
    print(f"VCM Rotational Vibration DOB: {max_error_dob:.6e}")
    
except Exception as e:
    print(f"Error in performance calculation: {e}")

plt.show()

