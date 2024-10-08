import numpy as np
import control as co
from scipy import signal

num_sector = 420                   # Number of sector
num_rpm = 7200                     # Number of RPM
Ts = 1/(num_rpm/60*num_sector)     # Sampling time
Mr_f = 2                           # Multi-rate number

# Plant parameters
# VCM parameters
Kp_vcm = 3.7976e+07
omega_vcm = list(np.array([0, 5300, 6100, 6500, 8050, 9600, 14800, 17400, 21000, 26000, 26600, 29000, 32200, 38300, 43300, 44800]) * 2 * np.pi)
kappa_vcm = [1, -1.0, +0.1, -0.1, 0.04, -0.7, -0.2, -1.0, +3.0, -3.2, 2.1, -1.5, +2.0, -0.2, +0.3, -0.5]
zeta_vcm = [0, 0.02, 0.04, 0.02, 0.01, 0.03, 0.01, 0.02, 0.02, 0.012, 0.007, 0.01, 0.03, 0.01, 0.01, 0.01]

# PZT parameters
omega_pzt = list(np.array([14800, 21500, 28000, 40200, 42050, 44400, 46500, 100000]) * 2 * np.pi)
kappa_pzt = [-0.005, -0.01, -0.1, +0.8, 0.3, -0.25, 0.3, 10.0]
zeta_pzt = [0.025, 0.03, 0.05, 0.008, 0.008, 0.01, 0.02, 0.3]

def create_system(omega, kappa, zeta, Kp=1, omega_factor=1, zeta_factor=1):
    """
    Create a system using the given parameters.

    Args:
        omega (list): List of natural frequencies.
        kappa (list): List of gain factors.
        zeta (list): List of damping ratios.
        Kp (float, optional): Gain constant. Defaults to 1.
        omega_factor (float, optional): Factor to scale natural frequencies. Defaults to 1.
        zeta_factor (float, optional): Factor to scale damping ratios. Defaults to 1.

    Returns:
        StateSpace: The created system as a StateSpace object.
    """

    Sys = 0
    for i in range(len(omega)):
        Sys_i = co.tf2ss(co.tf([0, 0, kappa[i] * Kp],
                               [1, 2 * zeta[i] * zeta_factor * omega[i] * omega_factor, (omega[i] * omega_factor) ** 2]))
        Sys = Sys + Sys_i
    return Sys

def normalize_pzt_system(Sys_pzt):
    """
    Normalize the PZT system based on its frequency response at zero frequency.

    Args:
        Sys_pzt (StateSpace): The PZT system to be normalized.

    Returns:
        StateSpace: The normalized PZT system.
    """
    Sys_pzt_ss = [Sys_pzt.A, Sys_pzt.B, Sys_pzt.C, Sys_pzt.D] 	
    _, pzt_freqresp = signal.freqresp(Sys_pzt_ss, np.array([0.]))
    Sys_pzt = Sys_pzt / abs(pzt_freqresp)
    return Sys_pzt

# LT (case 1)
Sys_Pc_vcm_c1 = create_system(omega_vcm, kappa_vcm, zeta_vcm, Kp_vcm, 1.04, 0.8)
Sys_Pc_pzt_c1 = create_system(omega_pzt, kappa_pzt, zeta_pzt, omega_factor=1.06, zeta_factor=0.8)
Sys_Pc_pzt_c1 = normalize_pzt_system(Sys_Pc_pzt_c1)

# RT (Case 2)
Sys_Pc_vcm_c2 = create_system(omega_vcm, kappa_vcm, zeta_vcm, Kp_vcm)
Sys_Pc_pzt_c2 = create_system(omega_pzt, kappa_pzt, zeta_pzt)
Sys_Pc_pzt_c2 = normalize_pzt_system(Sys_Pc_pzt_c2)

# HT (Case 3)
# VCM
Sys_Pc_vcm_c3 = create_system(omega_vcm, kappa_vcm, zeta_vcm, Kp_vcm, 0.96, 1.2)
Sys_Pc_pzt_c3 = create_system(omega_pzt, kappa_pzt, zeta_pzt, omega_factor=0.94, zeta_factor=1.2)
Sys_Pc_pzt_c3 = normalize_pzt_system(Sys_Pc_pzt_c3)


# LT / PZT gain +5% (Case 4)
Sys_Pc_vcm_c4 = Sys_Pc_vcm_c1
Sys_Pc_pzt_c4 = Sys_Pc_pzt_c1*1.05


# RT / PZT gain +5% (Case 5)
Sys_Pc_vcm_c5 = Sys_Pc_vcm_c2
Sys_Pc_pzt_c5 = Sys_Pc_pzt_c2*1.05


# HT / PZT gain +5% (Case 6)
Sys_Pc_vcm_c6 = Sys_Pc_vcm_c3
Sys_Pc_pzt_c6 = Sys_Pc_pzt_c3*1.05


# LT / PZT gain -5% (Case 7)
Sys_Pc_vcm_c7 = Sys_Pc_vcm_c1
Sys_Pc_pzt_c7 = Sys_Pc_pzt_c1*0.95


# RT / PZT gain -5% (Case 8)
Sys_Pc_vcm_c8 = Sys_Pc_vcm_c2
Sys_Pc_pzt_c8 = Sys_Pc_pzt_c2*0.95


# HT / PZT gain -5% (Case 9)
Sys_Pc_vcm_c9 = Sys_Pc_vcm_c3
Sys_Pc_pzt_c9 = Sys_Pc_pzt_c3*0.95


Sys_Pc_vcm_all = [Sys_Pc_vcm_c1, Sys_Pc_vcm_c2, Sys_Pc_vcm_c3, 
                Sys_Pc_vcm_c4, Sys_Pc_vcm_c5, Sys_Pc_vcm_c6, 
                Sys_Pc_vcm_c7, Sys_Pc_vcm_c8, Sys_Pc_vcm_c9]

Sys_Pc_pzt_all = [Sys_Pc_pzt_c1, Sys_Pc_pzt_c2, Sys_Pc_pzt_c3, 
                Sys_Pc_pzt_c4, Sys_Pc_pzt_c5, Sys_Pc_pzt_c6, 
                Sys_Pc_pzt_c7, Sys_Pc_pzt_c8, Sys_Pc_pzt_c9]

