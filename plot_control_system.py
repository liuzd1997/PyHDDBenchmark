import numpy as np
from utils import *
import plant
from control import matlab

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
    Sys_Pdm0_vcm = matlab.c2d(Sys_Pc_vcm, Ts/Mr_f, 'zoh')
    Sys_Pdm_vcm = Sys_Pdm0_vcm * Sys_Fm_vcm
    Sys_Pd_vcm = dts_resampling(Sys_Pdm_vcm, Mr_f)

    Sys_Pdm0_pzt = matlab.c2d(Sys_Pc_pzt, Ts/Mr_f, 'zoh')
    Sys_Pdm_pzt = Sys_Pdm0_pzt * Sys_Fm_pzt
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

# Frequency response
f = np.logspace(1, np.log10(60000), 3000)

Fr_Pc_vcm_all = freqresp(plant.Sys_Pc_vcm_all, f*2*np.pi) 
Fr_Pc_pzt_all = freqresp(plant.Sys_Pc_pzt_all, f*2*np.pi)
Fr_Pd_vcm_all = freqresp(Sys_Pd_vcm_all, f*2*np.pi)
Fr_Pd_pzt_all = freqresp(Sys_Pd_pzt_all, f*2*np.pi)
Fr_Cd_vcm = freqresp(Sys_Cd_vcm, f*2*np.pi)
Fr_Cd_pzt = freqresp(Sys_Cd_pzt, f*2*np.pi)
Fr_Fm_vcm = freqresp(Sys_Fm_vcm, f*2*np.pi)
Fr_Fm_pzt = freqresp(Sys_Fm_pzt, f*2*np.pi)

Fr_L_vcm_all = Fr_Pd_vcm_all * Fr_Cd_vcm
Fr_L_pzt_all = Fr_Pd_pzt_all * Fr_Cd_pzt

Fr_L = Fr_L_vcm_all + Fr_L_pzt_all
Fr_S = 1. / (1 + Fr_L)

# Plot the Frequency Response of the system 

# Fr_Pc_vcm_all
save_path = get_plot_path("figure9_The_Frequency_Response_of_Pc_vcm.png")
title = 'The Frequency Response of $Pc_{vcm}$' 
Fr_Pc_vcm_all_mag = 20*np.log10(abs(Fr_Pc_vcm_all).T)
Fr_Pc_vcm_all_phase = 180*(np.angle(Fr_Pc_vcm_all).T)/np.pi - 180
Freq_Resp_Plot(Fr_Pc_vcm_all_mag, Fr_Pc_vcm_all_phase, f, title, (-360,0), save_path)

# Fr_Pc_pzt_all
save_path = get_plot_path("figure10_The_Frequency_Response_of_Pc_pzt.png")
title = 'The Frequency Response of $Pc_{pzt}$' 
Fr_Pc_pzt_all_mag = 20*np.log10(abs(Fr_Pc_pzt_all).T)
Fr_Pc_pzt_all_phase = 180*(np.angle(Fr_Pc_pzt_all).T)/np.pi
Freq_Resp_Plot(Fr_Pc_pzt_all_mag, Fr_Pc_pzt_all_phase, f, title, (-180,180), save_path)

# Multi_Rate_Filter
save_path = get_plot_path("figure11_Multi-rate_filter.png")
Fr_Fm_vcm_mag = 20*np.log10(abs(Fr_Fm_vcm))
Fr_Fm_vcm_phase = 180*(np.angle(Fr_Fm_vcm))/np.pi
Fr_Fm_pzt_mag = 20*np.log10(abs(Fr_Fm_pzt))
Fr_Fm_pzt_phase = 180*(np.angle(Fr_Fm_pzt))/np.pi
Multi_Rate_Filter_Plot(Fr_Fm_vcm_mag, Fr_Fm_vcm_phase, Fr_Fm_pzt_mag, Fr_Fm_pzt_phase, f, 'Multi-rate filter', save_path)

# Fr_Pd_vcm_all
save_path = get_plot_path("figure12_The_Frequency_Response_of_Pd_vcm.png")
title = 'The Frequency Response of $Pd_{vcm}$' 
Fr_Pd_vcm_all_mag = 20*np.log10(abs(Fr_Pd_vcm_all).T)
Fr_Pd_vcm_all_phase = 180*(np.angle(Fr_Pd_vcm_all).T)/np.pi - 180
Freq_Resp_Plot(Fr_Pd_vcm_all_mag, Fr_Pd_vcm_all_phase, f, title, (-360,90), save_path)

# Fr_Pd_pzt_all
save_path = get_plot_path("figure13_The_Frequency_Response_of_Pd_pzt.png")
title = 'The Frequency Response of $Pd_{pzt}$' 
Fr_Pd_pzt_all_mag = 20*np.log10(abs(Fr_Pd_pzt_all).T)
Fr_Pd_pzt_all_phase = 180*(np.angle(Fr_Pd_pzt_all).T)/np.pi
Freq_Resp_Plot(Fr_Pd_pzt_all_mag, Fr_Pd_pzt_all_phase, f, title, (-180,180), save_path)

# Fr_Cd_vcm
save_path = get_plot_path("figure14_The_Frequency_Response_of_Cd_vcm.png")
title = 'The Frequency Response of $Cd_{vcm}$' 
Fr_Cd_vcm_mag = 20*np.log10(abs(Fr_Cd_vcm).T)
Fr_Cd_vcm_phase = 180*(np.angle(Fr_Cd_vcm).T)/np.pi
Freq_Resp_Plot(Fr_Cd_vcm_mag, Fr_Cd_vcm_phase, f, title, (-180, 180), save_path)

# Fr_Cd_pzt
save_path = get_plot_path("figure15_The_Frequency_Response_of_Cd_vcm.png")
title = 'The Frequency Response of $Cd_{vcm}$' 
Fr_Cd_pzt_mag = 20*np.log10(abs(Fr_Cd_pzt).T)
Fr_Cd_pzt_phase = 180*(np.angle(Fr_Cd_pzt).T)/np.pi
Freq_Resp_Plot(Fr_Cd_pzt_mag, Fr_Cd_pzt_phase, f, title, (-180, 180), save_path)

# Fr_L
save_path = get_plot_path("figure16_Openloop(Bode Plot).png")
title = 'Openloop (Bode Plot)' 
Fr_L_mag = 20*np.log10(abs(Fr_L).T)
Fr_L_phase = np.mod(180*(np.angle(Fr_L).T)/np.pi + 360, 360) - 360
Freq_Resp_Plot(Fr_L_mag, Fr_L_phase, f, title, (-360,0), save_path)

# Fr_L Nyquist Plot
save_path = get_plot_path("figure17_Openloop(Nyquist Plot).png")
title = 'Openloop(Nyquist Plot)'
for i in range(f.shape[0]):
    if f[i] > 1/Ts/2:
        index = i
        break 
Fr_L_real = np.real(((Fr_L).T)[:, :index]).tolist()
Fr_L_imag = np.imag(((Fr_L).T)[:, :index]).tolist()
Nyquist_Plot(Fr_L_real, Fr_L_imag, title, save_path)

# Fr_Pd_vcm_all
save_path = get_plot_path("figure18_Sensitive_Function.png")
title = 'Sensitive Function' 
Fr_S_mag = 20*np.log10(abs(Fr_S).T)
Sensitive_Function_Plot(Fr_S_mag, f, title, save_path)