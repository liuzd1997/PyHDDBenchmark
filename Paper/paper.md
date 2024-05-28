---
title: 'Magnetic-Head Positioning Control System in HDDs - Python Version'
tags:
  - Python
  - hard disk drive servo control
  - positioning control

authors:
  - name: 
    orcid: 
    affiliation: "1"

  - name: 
    orcid: 
    affiliation: "2"

affiliations:
 - name: Mechatronics, Automation, and Control Systems Laboratory, University of Washington, Seattle, USA
   index: 1
date: 1 May 2024
bibliography: paper.bib
---

# Summary
Data capacity of hard disk drives (HDDs) must increase to meet the growing need for data storage. As a result, improving the positioning accuracy of the magnetic head in the HDD is essential for meeting this demand. Simulation is a highly efficient method for studying HDD control systems and this project establishes a magnetic-head positioning control system for HDDs adapted from MATLAB into Python, including plants, feedback controllers and multi-rate filters for voice coil motors (VCMs) and PZT actuators, while considering the disturbances caused by repeatable run-out (RRO), rotational vibration (RV), and fan-induced vibration. This project also provides 9 built in cases to represent the different characteristics under varying temperature conditions (low temperature, room temperature, and high temperature) and different PZT actuator gains (± 5%).
The organization and content of this repository are inspired by the Matlab reference source [@Takenori2024Magnetic](https://www.mathworks.com/matlabcentral/fileexchange/111515-magnetic-head-positioning-control-system-in-hdds), and is designed to enhance the navigability and accessibility of Hard Disk Drive (HDD) benchmark problem simulations and their analyses.




# Statement of need
We translated MATLAB based benchmark problems (Magnetic-head positioning control system in HDDs - File Exchange - MATLAB Central (mathworks.com) into a Python-based package. Development of the package in Python extends the usage for simulating the magnetic-head positioning control system in the latest HDDs. The increasing data capacity requirements of hard disk drives motivates improvements in the positioning accuracy of the magnetic head for future advancements in data storage.

Previous research have leveraged the MATLAB version of the HDD package, including research conducted by [@muto2023controller](https://www.sciencedirect.com/science/article/pii/S2405896323022401), who proposed a recurrent neural netowrk based (RNN-based) reinforcement learning (RL) solution for HDD control. One of the methods in this study was to to transform the RNN-based controller into a state-space linear controller to ensure stability. Implementing the RNN-based RL solution improved the system performance by 5.8% compared to the original benchmark. [@yabui2023control](https://www.sciencedirect.com/science/article/pii/S2405896323011874) developed an adaptive feedforward cancellation (AFC) control to address repeatable runout (RRO) in the tracks of the disk. RRO has the potential to distort the track shape on the disks, affecting overall system performance. By implementing this control system, Yabui et all. independently controlled synchronous and asynchronous RRO, thereby eliminating any interdependencies that could potentially impact the system response. The AFC controller successfully increased the minimum distance in adjacent tracks by 2nm. [@ouyang2023recursive](https://doi.org/10.1115/1.4063481) utilized the HDD benchmark package as one of the applications in their study on system identification. Their approach focused on non-uniformly sampled system identification based on recursive least-squares (RLS) and coprime collaborative sensing. Results from this study demonstrate that the algorithm effectively tracks fast systems beyond the Nyquist frequencies of multiple slow sensors. 

These examples highlight the diverse applications and extensions of studies related to HDD control, indicating a growing demand for its utilization.

# Features & Functionality
This software is used to simulate the magnetic-head positioning control system. The magnetic head consists of a voice coil motors (VCM) and a PZT actuator, as shown in Figure \autoref{fig:HDD}. 

![HDD structure. \label{fig:HDD}](./Figures/HDD.jpg){width=60%}

Figure \autoref{fig:ControlBlockDiagram} shows the control block diagram of magnetic-head positioning control system, where $P_{cv}$ is the VCM in conyinuous-time system, $P_{cp}$ is the PZT actuator in continuous-time system, $C_{dv}$ is the feedback controller for VCM, $C_{dp}$ is the feedback controller for PZT actuator, $F_{mv}$ is the multi-rate filter for VCM, $F_{mp}$ is the multi-rate filter for PZT actuator, $I_p$ is the interpolator, $H_m$ is the multi-rate zero-order hold, $S$ is the samper, $d_p$ is the fan-induced vibration, $d_f$ is the rotational vibration, $d_{RRO}$ is the repeatable run-out (RRO), $y_c$ is the head position in continuous time, $y_d$ is the head position in descrete time, and $y_{cp} is the displacement of PZT actuator. As for the disturbances in the system, $d_{RRO}$ is the oscillation of target tracks written on the disk, $d_f$ is the external vibration caused by other HDDs in a storage box, $d_p$ is the external vibration caused by cooling fans in a storage box.

![Block diagram of magnetic-head positioning control system. \label{fig:ControlBlockDiagram}](./Figures/ControlBlockDiagram.jpg){width=60%}

A total of 9 cases of this system have been included for users to explore. 

| Case No. | 1          | 2          | 3          | 4          | 5          | 6          |      7     |      8     | 9          |
| :----:   | :----:     | :----:     | :----:     | :----:     | :----:     | :----:     | :----:     | :----:     | :----:     |
| Temp.    | LT         | RT         | HT         | LT         | RT         | HT         | LT         | RT         | HT         |
| PZT gain | Nominal    | Nominal    | Nominal    | Nominal+5% | Nominal+5% | Nominal+5% | Nominal-5% | Nominal-5% | Nominal-5% |

The controlled object has vatiations due to temperature dependencies of mechanical resonant frequencies:
- LT (low temperature): +4% VCM nominal values, +6% PZT nominal values.
- RT (room temperature): same as nominal models.
- HT (high temperature): -4% VCM nominal values, -6% PZT nominal values.

Users can create their own system by defining approximate continuous-time systems, examples being any of the 9 use cases, or adjust the VCM and PZT parameters. And the parameters of the nominal controlled object are shown in [@atsumi2019quadruple].



# Description of the software
- `plant.py` specifies the dynamics of the plant being simulated. 
- `function_simulation.py` executes HDD simulations based on scenarios defined in `plant.py` and saves the outputs to a designated folder. This process may be time-consuming. 
- `simulate_trackfollow.py` displays simulation outcomes, requiring prior generation of simulation result files, and it gives the results of amplitude specturm of $d_f$, $d_p$, and $d_{RRO}$, the output displacement $y_{cp}$ and $y_c$. 
- `plot_control_system.py` visualizes the frequency responses of the control system. 
- `utils.py` includes additional data definitions and utility functions supporting the simulations. 
- `Data_RRO.txt` stores RRO data for function simulation. 
- `Fre_Resp.json` contains frequency response data.

Some exmaple results are shown in Figures \autoref{fig:Pc_pzt} to \autoref{fig:dRRO}. Figure \autoref{fig:Pc_pzt} and Figure \autoref{fig:Pc_vcm} show the frequency response of PZT actuator and VCM, respectively. Figure \autoref{fig:Multi-rate_filter} presents the multi-rate filters of PZT actuator and VCM. Figure \autoref{fig:Amplitude_spectrum_of_yc} illustrates the amplitude spectrum of the head position. Figures \autoref{fig:Amplitude_spectrum_of_df} and \autoref{fig:Amplitude_spectrum_of_dp} display the amplitude spectrum of the rotational vibration and fan-induced vibration, respectively. Finally, Figure \autoref{fig:dRRO} shows the amplitude of the repeatable run-out.

![Frequency response of PZT actuator. \label{fig:Pc_pzt}](./Figures/Frequency_Response_of_Pc_pzt.png){width=60%}
![Frequency response of VCM. \label{fig:Pc_vcm}](./Figures/Frequency_Response_of_Pc_vcm.png){width=60%}
![Multi-rate filters of PZT actuator and VCM. \label{fig:Multi-rate_filter}](./Figures/Multi-rate_filter.png){width=60%}
![Amplitude spectrum of the head position. \label{fig:Amplitude_spectrum_of_yc}](./Figures/Amplitude_spectrum_of_yc.png){width=60%}
![Amplitude spectrum of the rotational vibration. \label{fig:Amplitude_spectrum_of_df}](./Figures/Amplitude_spectrum_of_df.png){width=60%}
![Amplitude spectrum of the fan-induced vibration. \label{fig:Amplitude_spectrum_of_dp}](./Figures/Amplitude_spectrum_of_dp.png){width=60%}
![Amplitude of the repeatable run-out. \label{fig:dRRO}](./Figures/dRRO.png){width=60%}

# Acknowledgements


# References
Takenori Atsumi (2024). Magnetic-head positioning control system in HDDs (https://www.mathworks.com/matlabcentral/fileexchange/111515-magnetic-head-positioning-control-system-in-hdds), MATLAB Central File Exchange. Retrieved May 22, 2024.

R. Muto and Y. Uchimura, ``Controller design for hdd benchmark problem using rnn-based reinforcement learning,'' in The 22nd IFAC World Congress 2023, pp. 4830-4835, 2023.

S.Yabui, TAtsumi, and A.Okuyama, ``Control scheme of rro compensation for track mis-registration in hdds,'' in The 22nd IFAC World Congress 2023, pp. 6905-6910, 2023. 

Ouyang, J., and Chen, X. (October 25, 2023). "A Recursive System Identification With Non-Uniform Temporal Feedback Under Coprime Collaborative Sensing." ASME. Letters Dyn. Sys. Control. April 2023; 3(2): 021010. https://doi.org/10.1115/1.4063481
