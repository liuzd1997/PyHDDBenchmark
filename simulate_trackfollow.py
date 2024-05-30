import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils
import plant
import sys

Tp = 5.2697e-8  # 482 kTPI
Ts = plant.Ts  # get Ts from the plant parameters

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


# Figure 1: Amplitude spectrum of df
plt.figure(1)
plt.semilogx(sim_results[0]["freq"], 20 * np.log10(np.abs(sim_results[0]["Fr_df"])))
plt.title('Amplitude spectrum of $d_f$')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.xlim([10, 1/Ts/2])
plt.savefig(utils.get_plot_path("figure1_Amplitude_spectrum_of_df.png"))

# Figure 2: Amplitude spectrum of dp
plt.figure(2)
plt.semilogx(sim_results[0]["freq"], 20 * np.log10(np.abs(sim_results[0]["Fr_dp"])))
plt.title('Amplitude spectrum of $d_p$')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.xlim([10, 1/Ts/2])
plt.savefig(utils.get_plot_path("figure2_Amplitude_spectrum_of_dp.png"))


# Figure 3: dRRO
plt.figure(3)
plt.plot(sim_results[0]["time"][1:420*20] * 1e3, sim_results[0]["dRRO"][1:420*20] * 1e9)
plt.title('$d_{RRO}$')
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [nm]')
plt.grid(True)
plt.xlim([0, Ts*420*1e3])
plt.savefig(utils.get_plot_path("figure3_dRRO.png"))


# Figure 4: ypc
plt.figure(4)
for i, sim_result in enumerate(sim_results):
    plt.plot(sim_result["time"] * 1e3, sim_result["yc_pzt"] * 1e9, '--' if i >= 7 else '-')
plt.title('$y_{pc}$')
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [nm]')
plt.grid(True)
plt.legend([f'Case {i+1}' for i in range(9)], loc="lower left")
plt.savefig(utils.get_plot_path("figure4_ypc.png"))


# Figure 5: Max of |ycp|
plt.figure(5)
res = np.array([sim_result["yc_pzt"] for sim_result in sim_results]).transpose()
plt.plot(np.arange(1, 10), 1e9 * np.max(np.abs(res), axis=0), 'o')
plt.title('Max of $|y_{cp}|$')
plt.xlabel('Case number')
plt.ylabel('Value [nm]')
plt.grid(True)
plt.savefig(utils.get_plot_path("figure5_Max_of_abs(ycp).png"))

# Figure 6: yc
plt.figure(6)
for i, sim_result in enumerate(sim_results):
    plt.plot(sim_result["time"] * 1e3, sim_result["yc"] * 1e9, '--' if i >= 7 else '-')
plt.title('$y_c$')
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [% of Track width]')
plt.grid(True)
plt.legend([f'Case {i+1}' for i in range(9)], loc="lower left")
plt.savefig(utils.get_plot_path("figure6_yc.png"))

# Figure 7: Amplitude spectrum of yc
plt.figure(7)
for i, sim_result in enumerate(sim_results):
    plt.semilogx(sim_result["freq"], 20 * np.log10(np.abs(sim_result["Fr_yc"])), '--' if i >= 7 else '-')
plt.title('Amplitude spectrum of $y_c$')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.xlim([10, 50e3])
plt.legend([f'Case {i+1}' for i in range(9)], loc="lower left")
plt.savefig(utils.get_plot_path("figure7_Amplitude_spectrum_of_yc.png"))

# Figure 8: 3 sigma of yc
plt.figure(8)
res = 3 * np.std(np.array([sim_result["yc"] for sim_result in sim_results]), axis=1).transpose()
plt.plot(np.arange(1, 10), res / Tp * 100, 'o')
plt.title('3 sigma of $y_c$')
plt.xlabel('Case number')
plt.ylabel('Value [% of Track width]')
plt.grid(True)
plt.savefig(utils.get_plot_path("figure8_3_sigma_of_yc.png"))
