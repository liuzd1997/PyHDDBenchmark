---
title: 'Magnetic-Head Positioning Control System in HDDs - Python Version'
tags:
  - Python
  - dard disk drive servo control
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
The HDD Python package is ...
Derived from Atsumi's work in MATLAB ...
Changes made in Python due to ...



# Statement of need
We translated MATLAB based benchmark problems (Magnetic-head positioning control system in HDDs - File Exchange - MATLAB Central (mathworks.com)) into a Python-based package. Development of the package in Python extends the usage for simulating the magnetic-head positioning control system in the latest HDDs. The increasing data capacity requirements of hard disk drives motivates improvements in the positioning accuracy of the magnetic head for future advancements in data storage.

Previous research have leveraged the MATLAB version of the HDD package, including research conducted by  [@muto2023controller], who proposed a recurrent neural netowrk based (RNN-based) reinforcement learning (RL) solution for HDD control. One of the methods in this study was to to transform the RNN-based controller into a state-space linear controller to ensure stability. Implementing the RNN-based RL solution improved the system performance by 5.8% compared to the original benchmark. [@yabui2023control] developed an adaptive feedforward cancellation (AFC) control to address repeatable runout (RRO) in the tracks of the disk. RRO has the potential to distort the track shape on the disks, affecting overall system performance. By implementing this control system, Yabui et all. independently controlled synchronous and asynchronous RRO, thereby eliminating any interdependencies that could potentially impact the system response. The AFC controller successfully increased the minimum distance in adjacent tracks by 2nm. [@ouyang2023recursive] utilized the HDD benchmark package as one of the applications in their study on system identification. Their approach focused on non-uniformly sampled system identification based on recursive least-squares (RLS) and coprime collaborative sensing. Results from this study demonstrate that the algorithm effectively tracks fast systems beyond the Nyquist frequencies of multiple slow sensors. 

These examples highlight the diverse applications and extensions of studies related to HDD control, indicating a growing demand for its utilization.

# Features & Functionality
This software is used to simulate the magnetic-head positioning control system.  The magnetic head consists of a voice coil motors (VCM) and a PZT actuator. Figure \ref{fig:ControlBlockDiagram} shows the control block diagram of magnetic-head positioning control system, where $P_{cv}$ is the VCM in conyinuous-time system, $P_{cp}$ is the PZT actuator in continuous-time system, $C_{dv}$ is the feedback controller for VCM, $C_{dp}$ is the feedback controller for PZT actuator, $F_{mv}$ is the multi-rate filter for VCM, $F_{mp}$ is the multi-rate filter for PZT actuator, $I_p$ is the interpolator, $H_m$ is the multi-rate zero-order hold, $S$ is the samper, $d_p$ is the fan-induced vibration, $d_f$ is the rotational vibration, $d_{PRO}$ is the repeatable run-out (PRO), $y_c$ is the head position in continuous time, $y_d$ is the head position in descrete time, and $y_{cp} is the displacement of PZT actuator.  

\begin{figure}
\centering
\includegraphics[width=0.8\textwidth]{./Figures/ControlBlockDiagram.jpg}
\caption{Block diagram of magnetic-head positioning control system.}
\label{fig:ControlBlockDiagram}
\end{figure}


# Example Use Cases
A selection of system models has been made available for reference. These models can be found in Plant.py, in which the system is used in Plot_Control_System.py and Function_Simulation.py. A total of 9 cases of this system have been included for users to explore. Users can create their own system by defining approximate continuous-time systems, examples being any of the 9 use cases, or adjust the VCM and PZT parameters as indicated in the subsection ‘Plant parameter’.

Parameters of the nominal controlled object are shown in the following paper.
T. Atsumi and S. Yabui, “Quadruple-Stage Actuator System for Magnetic-Head Positioning System in HDDs,”
The IEEE Transactions on Industrial Electronics, Vol. 67, No. 11, pp. 9184-9194, (2020-11)

The use cases in this example system analyze temperature dependencies of mechanical resonant frequencies. They are summarized as follows:
Case 1: LT (low temperature), +4% VCM nominal values, +6% PZT nominal values
Case 2: RT (room temperature)
Case 3: HT (high temperature), -4% VCM nominal values, -6% PZT nominal values
Case 4: +5% Case 1 nominal values
Case 5: +5% Case 2 nominal values
Case 6: +5% Case 3 nominal values
Case 7: -5% Case 1 nominal values
Case 8: -5% Case 2 nominal values
Case 9: -5% Case 3 nominal values

Numerical data from the example system include: 
Data_RRO.txt: d_RRO（Repeatable Run-Out), oscillation of target tracks written on the disk.
Data_Cd: Feedback controllers. Data is translated from mat file Data_Cd.mat. 
In Tools.py, functions get_Sys_Cd_vcm and get_Sys_Cd_pzt.
In Function_Simulation.py, defined as variables Sys_Cd_vcm and Sys_Cd_pzt.
Data_Fm: Multi-rate filters. Data is translated from mat file Data_Fm.mat. 
In Tools.py, functions get_Sys_Fm_vcm and get_Sys_Fm_pzt.
In Function_Simulation.py, defined as variables Sys_Fm_vcm and Sys_Fm_pzt.

# Acknowledgements


# References
