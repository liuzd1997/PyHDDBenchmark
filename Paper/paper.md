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

# Background
The data capacity of the hard disk drive (HDD) must increase to meet the demands for data storage. As a result, we must improve the positioning accuracy of the magnetic head in the HDD for a better future. 

# Statement of need
We translate MATLAB based benchmark problems (Magnetic-head positioning control system in HDDs - File Exchange - MATLAB Central (mathworks.com)) into Python based programs. Now we can use Python to simulate the magnetic-head positioning control system used in the latest HDDs.
Developed on: Python 3.10.0
Packages: numpy 1.23.4, control 0.9.4, scipy 1.11.3, matplotlib 3.7.0

# Features & Functionality
will fill in as code is near completion

# Function Description?

# Dependencies and Environment Setup
## Environment Requirements
This project is compatible with Python 3.11.0 and was tested with specific package versions:
* numpy 1.23.4
* control 0.9.4
* scipy 1.11.3
* matplotlib 3.7.0
## Setting up the Environment
Follow these steps to prepare the environment for running the simulations and analyses:
1. Download and Unzip:
  * Download the code package and extract its contents.
1. Install Dependencies:
  * Use pip to install the necessary Python packages:<br>
  ````pip install -r requirements.txt````

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
