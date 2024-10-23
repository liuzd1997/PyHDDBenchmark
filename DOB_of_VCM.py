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
# prints ASCII art of the system
pa.print_system()

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

# Figure 20: dff
plt.figure(20)
# translate to ms and nm
plt.plot(sim_results[0]["time"][1:420*20] * 1e3, sim_results[0]["df"][1:420*20] * 1e9)
plt.title('$d_f$')
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [nm]')
plt.grid(True)
plt.xlim([0, Ts*420*1e3])
plt.savefig(utils.get_plot_path("figure20_df.png"))

length_of_slice = len(sim_results[0]["time"][1:420*20])
print("Length of sim_results[0]['time'][1:420*20]:", length_of_slice)

print("Ts*420*1e3:", Ts*420*1e3)




print("Discrete-time Plant:", Sys_Pd_vcm)
P_nom = Sys_Pd_vcm
cutoff_freq = 5
w = 2 * np.pi * cutoff_freq
low_pass_filter = signal.TransferFunction([w], [1, w])
low_pass_filter_disc = signal.cont2discrete(([w], [1, w]), Ts, method='zoh')
#low_pass_filter_disc = matlab.c2d(low_pass_filter, Ts, method='zoh')

# Extract the numerator and denominator from the result of cont2discrete
num_disc = low_pass_filter_disc[0].flatten()
den_disc = low_pass_filter_disc[1].flatten()

# Create a control.TransferFunction for the discrete-time system
low_pass_filter_tf = co.TransferFunction(num_disc, den_disc, Ts)
print("Discrete-time Low-pass Filter for DOB:", low_pass_filter_tf)


#P_nom_tf = co.ss2tf(P_nom)

# Access the numerator and denominator from the transfer function
#numerator = P_nom_tf.num[0][0]  # Extract the first (and usually only) element
#denominator = P_nom_tf.den[0][0]
#P_nom_inv = co.TransferFunction(denominator, numerator, Ts)  # 求名义模型的逆
#print("Inverse of Nominal Model:", P_nom_inv)


def get_regularized_inverse(P_nom, cutoff_freq):
    # Convert the nominal plant to transfer function if it's not already
    P_nom_tf = co.ss2tf(P_nom)

    # Extract numerator and denominator
    numerator = P_nom_tf.num[0][0]
    denominator = P_nom_tf.den[0][0]

    # Inverse of P_nom is simply flipping num and den
    P_nom_inv = co.TransferFunction(denominator, numerator, Ts)

    # Create low-pass filter to stabilize the inverse
    w = 2 * np.pi * cutoff_freq
    low_pass_filter = co.TransferFunction([w], [1, w], Ts)

    # Multiply the inverse by the low-pass filter to regularize it
    P_nom_inv_regularized = P_nom_inv * low_pass_filter

    return P_nom_inv_regularized



# Disturbance observer design

def disturbance_observer(u, y, P_nom, P_nom_inv, low_pass_filter_tf):
    # Estimating disturbance
    # Predict output based on nominal model
    y_nom = co.forced_response(P_nom, U=u, X0=0)[1]
    #print("y_nom=",y_nom)
    # Error between real and predicted output
    error = y - y_nom
    #print("error=",error)

    error_inv = co.forced_response(P_nom_inv, U=error, X0=0)[1]

    if np.any(np.isnan(error_inv)) or np.any(np.isinf(error_inv)):
        print("Warning: NaN or Inf detected in error_decoupled. Applying filtering.")
        error_inv = np.nan_to_num(error_inv) 

    disturbance_initial = error_inv-u
    # Estimate disturbance using low-pass filter
    disturbance_estimated = co.forced_response(low_pass_filter_tf, U=disturbance_initial, X0=0)[1]
    
    return disturbance_estimated


# Simulate a simple disturbance rejection
time = np.arange(0, Ts*420, Ts)  # Time vector


print("time:", Ts*420)
length_of_time = len(time)
print("Length of time:", length_of_time)

u = np.ones_like(time) * 1  # Step input

length_of_u = len(u)
print("Length of u:", length_of_u)
#desampling df, because the real time doesn't match with the length of time data point, which differ by 20 times

df_data = sim_results[0]["df"][0:420*20]
print(df_data.shape)
df = df_data[::20]
u_with_dis=df + u

#u_with_dis=sim_results[0]["df"][0:420*20] + u

y_step_response = co.forced_response(Sys_Pd_vcm, T=time, U=u)[1]

#print("dis=",sim_results[0]["df"][0:420])
#print("u=",u)
#print("u_with_dis=",u_with_dis)
#print("u_with_dis-u=",u_with_dis-u)
# Simulated output (for simplicity, assume this is measured)
y = co.forced_response(Sys_Pd_vcm, T=time, U=u_with_dis)[1]

plt.figure(23)

plt.plot(time * 1e3, y_step_response, label="Step response of VCM")
plt.plot(time * 1e3, y, label="Step response of VCM with disturbance", linestyle='--')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.title('Step response of VCM and VCM with disturbance')
plt.show()


P_nom_inv_regularized = get_regularized_inverse(P_nom, 5)
print("P_nom_inv_regularized:", P_nom_inv_regularized)
#print("y=",y)
# Apply disturbance observer
disturbance_estimated = disturbance_observer(u, y, Sys_Pd_vcm, P_nom_inv_regularized, low_pass_filter_tf)

# Plot the results
plt.figure()
# plt.plot(time, y, label="System Output")
plt.plot(time * 1e3, disturbance_estimated * 1e9, label="Estimated Disturbance")
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (nm)')
plt.legend()
plt.grid(True)
plt.title('Disturbance Estimation using DOB')
plt.savefig(utils.get_plot_path("figure19_The_Disturbance_Observer_of_Pc_vcm.png"))
plt.show()

