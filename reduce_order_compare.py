import control as co
from control import matlab
import numpy as np
from scipy import signal
from pdb import set_trace as pdb
import matplotlib.pyplot as plt
import json


def Freq_Resp_Plot_Compare(Fr_Resp_reduced_Mag, 
                           Fr_Resp_reduced_Phase, 
                           Fr_Resp_all_Mag, 
                           Fr_Resp_all_Phase, 
                           Freq, 
                           name, 
                           phase_range=(-360, 90), save_path=None):

    title = name

    fig, ax = plt.subplots(4,1, sharex='col', figsize = (24, 24))
    fig.suptitle(title, fontsize=22, weight='bold', family='Times New Roman')
    plt.xticks(font={'family': 'Times New Roman', 'size' : 16, 'weight': 'bold'})
    plt.yticks(font={'family': 'Times New Roman', 'size' : 16, 'weight': 'bold'})
    plt.xscale('log')
    
    l1 = []
    l2 = []
    label = []

    for i in range(Fr_Resp_reduced_Mag.shape[0]):

        Fr_Resp_Mag = Fr_Resp_reduced_Mag[i]
        Fr_Resp_Phase = Fr_Resp_reduced_Phase[i]

        # The setting of subfigure 1
        ax0 = ax[0]
        ax0.set_title('The result of reduced-order system', fontsize=18, weight='bold', family='Times New Roman')
        ax0.set_ylabel("Gain [dB]", fontdict={'family': 'Times New Roman',
                                                'size' : 18, 'weight': 'bold'}) 
        
        plt.sca(ax[0])
        plt.yticks(font={'family': 'Times New Roman', 'size' : 16, 'weight': 'bold'})
        
        # The setting of subfigure 2
        ax1 = ax[1]
        
        ax1.set_xlabel("Frequency [Hz]", fontdict={'family': 'Times New Roman',
                                                'size' : 18, 'weight': 'bold'}) 
        ax1.set_ylabel("Phase [deg.]", fontdict={'family': 'Times New Roman',
                                                'size' : 18, 'weight': 'bold'})
        
        plt.sca(ax[1])
        y_major_locator = plt.MultipleLocator(90)
        ax1.yaxis.set_major_locator(y_major_locator)
        plt.ylim(phase_range[0], phase_range[1])

        if i > 5:

            l, = ax0.plot(Freq, Fr_Resp_Mag, linestyle="--")
            l1.append(l)

            l, = ax1.plot(Freq, Fr_Resp_Phase, linestyle="--")
            l2.append(l)

        else:

            l, = ax0.plot(Freq, Fr_Resp_Mag, linestyle="-")
            l1.append(l)

            l, = ax1.plot(Freq, Fr_Resp_Phase, linestyle="-")
            l2.append(l)
        
        label.append('Case '+ str(i+1))
    if len(l2)>1:  
        ax1.legend(handles=l2, 
                labels=label,
                loc="lower left", 
                prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
                )

    l1 = []
    l2 = []
    label = []

    for i in range(Fr_Resp_all_Mag.shape[0]):

        Fr_Resp_Mag = Fr_Resp_all_Mag[i]
        Fr_Resp_Phase = Fr_Resp_all_Phase[i]

        # The setting of subfigure 1
        ax2 = ax[2]
        ax2.set_title('The result of full-order system', fontsize=18, weight='bold', family='Times New Roman')

        ax2.set_ylabel("Gain [dB]", fontdict={'family': 'Times New Roman',
                                                'size' : 18, 'weight': 'bold'}) 
        
        plt.sca(ax[2])
        plt.yticks(font={'family': 'Times New Roman', 'size' : 16, 'weight': 'bold'})
        
        # The setting of subfigure 2
        ax3 = ax[3]
        
        ax3.set_xlabel("Frequency [Hz]", fontdict={'family': 'Times New Roman',
                                                'size' : 18, 'weight': 'bold'}) 
        ax3.set_ylabel("Phase [deg.]", fontdict={'family': 'Times New Roman',
                                                'size' : 18, 'weight': 'bold'})
        
        plt.sca(ax[3])
        y_major_locator = plt.MultipleLocator(90)
        ax1.yaxis.set_major_locator(y_major_locator)
        plt.ylim(phase_range[0], phase_range[1])

        if i > 5:

            l, = ax2.plot(Freq, Fr_Resp_Mag, linestyle="--")
            l1.append(l)

            l, = ax3.plot(Freq, Fr_Resp_Phase, linestyle="--")
            l2.append(l)

        else:

            l, = ax2.plot(Freq, Fr_Resp_Mag, linestyle="-")
            l1.append(l)

            l, = ax3.plot(Freq, Fr_Resp_Phase, linestyle="-")
            l2.append(l)
        
        label.append('Case '+ str(i+1))
    if len(l2)>1:  
        ax3.legend(handles=l2, 
                labels=label,
                loc="lower left", 
                prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
                )
    
    if save_path != None:
        plt.savefig(save_path)


def Nyquist_Plot_Compare(Fr_Resp_all_real_reduce, 
                         Fr_Resp_all_imag_reduce,
                         Fr_Resp_all_real, 
                         Fr_Resp_all_imag, 
                         title, 
                         save_path=None):

    fig, ax = plt.subplots(1,4, sharex='col', figsize = (48, 24))
    fig.suptitle(title, fontsize=22, weight='bold', family='Times New Roman')
    plt.xticks(font={'family': 'Times New Roman', 'size' : 16, 'weight': 'bold'})
    plt.yticks(font={'family': 'Times New Roman', 'size' : 16, 'weight': 'bold'})

    ax0 = ax[0]
    ax0.set_title('Openloop (Nyquist Plot) Overall (Reduced Order)', fontsize=22, weight='bold', family='Times New Roman')
    ax0.set_xlabel("Real Axis", fontdict={'family': 'Times New Roman',
                                          'size' : 18, 'weight': 'bold'}) 
    ax0.set_ylabel("Imaginary Axis", fontdict={'family': 'Times New Roman',
                                               'size' : 18, 'weight': 'bold'})
    
    ax1 = ax[1]
    ax1.set_title('Openloop (Nyquist Plot) Detail (Reduced Order)', fontsize=22, weight='bold', family='Times New Roman')
    ax1.set_xlabel("Real Axis", fontdict={'family': 'Times New Roman',
                                          'size' : 18, 'weight': 'bold'}) 
    ax1.set_ylabel("Imaginary Axis", fontdict={'family': 'Times New Roman',
                                               'size' : 18, 'weight': 'bold'})
    plt.sca(ax[1])
    x_major_locator = plt.MultipleLocator(2)
    ax1.xaxis.set_major_locator(x_major_locator)
    plt.xlim(-7, 7)

    y_major_locator = plt.MultipleLocator(2)
    ax1.yaxis.set_major_locator(y_major_locator)
    plt.ylim(-5, 5)

    L1 = []
    L2 = []
    label = []

    for i in range(len(Fr_Resp_all_real_reduce)):

        Fr_Resp_real_all_reduce = Fr_Resp_all_real_reduce[i]
        Fr_Resp_imag_all_reduce = Fr_Resp_all_imag_reduce[i]
        
        if i > 5:
            l, = ax0.plot(Fr_Resp_real_all_reduce, Fr_Resp_imag_all_reduce, linestyle="--")
            L1.append(l)

        else:
            l, = ax0.plot(Fr_Resp_real_all_reduce, Fr_Resp_imag_all_reduce, linestyle="-")
            L1.append(l)
        
        for j in range(1, len(Fr_Resp_real_all_reduce)):
            if abs(Fr_Resp_real_all_reduce[-j]) > 7 or abs(Fr_Resp_imag_all_reduce[-j]) > 5:
                d_index = j
                break
        
        Fr_Resp_real_detail_reduce = Fr_Resp_real_all_reduce[-d_index:]
        Fr_Resp_imag_detail_reduce = Fr_Resp_imag_all_reduce[-d_index:]

        if i > 5:
            l, = ax1.plot(Fr_Resp_real_detail_reduce, Fr_Resp_imag_detail_reduce, linestyle="--")
            L2.append(l)

        else:
            l, = ax1.plot(Fr_Resp_real_detail_reduce, Fr_Resp_imag_detail_reduce, linestyle="-")
            L2.append(l)
        
        label.append('Case '+ str(i+1))
        
    ax0.legend(handles=L1, 
               labels=label,
               loc="lower left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
            )

    ax1.legend(handles=L2, 
               labels=label,
               loc="lower left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
            ) 
    
    ax2 = ax[2]
    ax2.set_title('Openloop (Nyquist Plot) Overall (Full Order)', fontsize=22, weight='bold', family='Times New Roman')
    ax2.set_xlabel("Real Axis", fontdict={'family': 'Times New Roman',
                                          'size' : 18, 'weight': 'bold'}) 
    ax2.set_ylabel("Imaginary Axis", fontdict={'family': 'Times New Roman',
                                               'size' : 18, 'weight': 'bold'})
    
    ax3 = ax[3]
    ax3.set_title('Openloop (Nyquist Plot) Detail (Full Order)', fontsize=22, weight='bold', family='Times New Roman')
    ax3.set_xlabel("Real Axis", fontdict={'family': 'Times New Roman',
                                          'size' : 18, 'weight': 'bold'}) 
    ax3.set_ylabel("Imaginary Axis", fontdict={'family': 'Times New Roman',
                                               'size' : 18, 'weight': 'bold'})
    plt.sca(ax[3])
    x_major_locator = plt.MultipleLocator(2)
    ax3.xaxis.set_major_locator(x_major_locator)
    plt.xlim(-7, 7)

    y_major_locator = plt.MultipleLocator(2)
    ax3.yaxis.set_major_locator(y_major_locator)
    plt.ylim(-5, 5)

    L1 = []
    L2 = []
    label = []

    for i in range(len(Fr_Resp_all_real)):

        Fr_Resp_real_all = Fr_Resp_all_real[i]
        Fr_Resp_imag_all = Fr_Resp_all_imag[i]
        
        if i > 5:
            l, = ax2.plot(Fr_Resp_real_all, Fr_Resp_imag_all, linestyle="--")
            L1.append(l)

        else:
            l, = ax2.plot(Fr_Resp_real_all, Fr_Resp_imag_all, linestyle="-")
            L1.append(l)
        
        for j in range(1, len(Fr_Resp_real_all)):
            if abs(Fr_Resp_real_all[-j]) > 7 or abs(Fr_Resp_imag_all[-j]) > 5:
                d_index = j
                break
        
        Fr_Resp_real_detail = Fr_Resp_real_all[-d_index:]
        Fr_Resp_imag_detail = Fr_Resp_imag_all[-d_index:]

        if i > 5:
            l, = ax3.plot(Fr_Resp_real_detail, Fr_Resp_imag_detail, linestyle="--")
            L2.append(l)

        else:
            l, = ax3.plot(Fr_Resp_real_detail, Fr_Resp_imag_detail, linestyle="-")
            L2.append(l)
        
        label.append('Case '+ str(i+1))
        
    ax2.legend(handles=L1, 
               labels=label,
               loc="lower left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
            )

    ax3.legend(handles=L2, 
               labels=label,
               loc="lower left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
            )
    
    if save_path != None:
        plt.savefig(save_path)


def Sensitive_Function_Plot_Compare(Fr_Resp_all_Mag_reduce, 
                                    Fr_Resp_all_Mag, 
                                    Freq, name, save_path=None):

    title = name

    fig, ax = plt.subplots(2,1, figsize = (16, 12))
    fig.suptitle(title, fontsize=22, weight='bold', family='Times New Roman')
    plt.xticks(font={'family': 'Times New Roman', 'size' : 16, 'weight': 'bold'})
    plt.yticks(font={'family': 'Times New Roman', 'size' : 16, 'weight': 'bold'})
    plt.xscale('log')
    
    l1 = []
    label = []
    ax0 = ax[0]

    for i in range(Fr_Resp_all_Mag_reduce.shape[0]):

        Fr_Resp_Mag_reduce = Fr_Resp_all_Mag_reduce[i]        
        ax0.set_xlabel("Frequency [Hz]", fontdict={'family': 'Times New Roman',
                                                'size' : 18, 'weight': 'bold'}) 
        ax0.set_ylabel("Gain [dB]", fontdict={'family': 'Times New Roman',
                                                'size' : 18, 'weight': 'bold'}) 

        if i > 5:
            l, = ax0.plot(Freq, Fr_Resp_Mag_reduce, linestyle="--")
            l1.append(l)

        else:
            l, = ax0.plot(Freq, Fr_Resp_Mag_reduce, linestyle="-")
            l1.append(l)

        label.append('Case '+ str(i+1))
        
    ax0.legend(handles=l1, 
               labels=label,
               loc="upper left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
            )
    
    l1 = []
    label = []
    ax1 = ax[1]

    for i in range(Fr_Resp_all_Mag.shape[0]):

        Fr_Resp_Mag = Fr_Resp_all_Mag[i]        
        ax1.set_xlabel("Frequency [Hz]", fontdict={'family': 'Times New Roman',
                                                'size' : 18, 'weight': 'bold'}) 
        ax1.set_ylabel("Gain [dB]", fontdict={'family': 'Times New Roman',
                                                'size' : 18, 'weight': 'bold'}) 

        if i > 5:
            l, = ax1.plot(Freq, Fr_Resp_Mag, linestyle="--")
            l1.append(l)

        else:
            l, = ax1.plot(Freq, Fr_Resp_Mag, linestyle="-")
            l1.append(l)

        label.append('Case '+ str(i+1))
        
    ax1.legend(handles=l1, 
               labels=label,
               loc="upper left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
            )
    
    if save_path != None:
        plt.savefig(save_path)


def Multi_Rate_Filter_Plot_Compare(Fr_Resp_1_Mag_reduce, 
                           Fr_Resp_1_Phase_reduce, 
                           Fr_Resp_2_Mag_reduce, 
                           Fr_Resp_2_Phase_reduce,
                           Fr_Resp_1_Mag, 
                           Fr_Resp_1_Phase, 
                           Fr_Resp_2_Mag, 
                           Fr_Resp_2_Phase, 
                           Freq, name, save_path=None):

    title = name

    fig, ax = plt.subplots(4,1, sharex='col', figsize = (16, 24))
    fig.suptitle(title, fontsize=22, weight='bold', family='Times New Roman')
    plt.xticks(font={'family': 'Times New Roman', 'size' : 16, 'weight': 'bold'})
    plt.yticks(font={'family': 'Times New Roman', 'size' : 16, 'weight': 'bold'})
    plt.xscale('log')
    l1 = []
    l2 = []
    label = ['Fr_Fm_vcm', 'Fr_Fm_pzt']

    # The setting of subfigure 1
    ax0 = ax[0]
    
    ax0.set_xlabel("Frequency [Hz]", fontdict={'family': 'Times New Roman',
                                            'size' : 18, 'weight': 'bold'}) 
    ax0.set_ylabel("Gain [dB]", fontdict={'family': 'Times New Roman',
                                            'size' : 18, 'weight': 'bold'}) 
    
    plt.sca(ax[0])
    plt.yticks(font={'family': 'Times New Roman', 'size' : 16, 'weight': 'bold'})

    l, = ax0.plot(Freq, Fr_Resp_1_Mag_reduce)
    l1.append(l)
    l, = ax0.plot(Freq, Fr_Resp_2_Mag_reduce)
    l1.append(l)

    ax0.legend(handles=l1, 
               labels=label,
               loc="lower left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
               )
    
    # The setting of subfigure 2
    ax1 = ax[1]
    
    ax1.set_xlabel("Frequency [Hz]", fontdict={'family': 'Times New Roman',
                                            'size' : 18, 'weight': 'bold'}) 
    ax1.set_ylabel("Phase [deg.]", fontdict={'family': 'Times New Roman',
                                            'size' : 18, 'weight': 'bold'})
    
    plt.sca(ax[1])
    y_major_locator = plt.MultipleLocator(90)
    ax1.yaxis.set_major_locator(y_major_locator)
    plt.ylim(-180, 180)

    l, = ax1.plot(Freq, Fr_Resp_1_Phase_reduce)
    l1.append(l)
    l, = ax1.plot(Freq, Fr_Resp_2_Phase_reduce)
    l1.append(l)

    ax1.legend(handles=l1, 
               labels=label,
               loc="lower left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
               )
    

    l1 = []
    l2 = []
    label = ['Fr_Fm_vcm', 'Fr_Fm_pzt']

    # The setting of subfigure 1
    ax2 = ax[2]
    
    ax2.set_xlabel("Frequency [Hz]", fontdict={'family': 'Times New Roman',
                                            'size' : 18, 'weight': 'bold'}) 
    ax2.set_ylabel("Gain [dB]", fontdict={'family': 'Times New Roman',
                                            'size' : 18, 'weight': 'bold'}) 
    
    plt.sca(ax[2])
    plt.yticks(font={'family': 'Times New Roman', 'size' : 16, 'weight': 'bold'})

    l, = ax2.plot(Freq, Fr_Resp_1_Mag)
    l1.append(l)
    l, = ax2.plot(Freq, Fr_Resp_2_Mag)
    l1.append(l)

    ax2.legend(handles=l1, 
               labels=label,
               loc="lower left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
               )
    
    # The setting of subfigure 2
    ax3 = ax[3]
    
    ax3.set_xlabel("Frequency [Hz]", fontdict={'family': 'Times New Roman',
                                            'size' : 18, 'weight': 'bold'}) 
    ax3.set_ylabel("Phase [deg.]", fontdict={'family': 'Times New Roman',
                                            'size' : 18, 'weight': 'bold'})
    
    plt.sca(ax[3])
    y_major_locator = plt.MultipleLocator(90)
    ax3.yaxis.set_major_locator(y_major_locator)
    plt.ylim(-180, 180)

    l, = ax3.plot(Freq, Fr_Resp_1_Phase)
    l1.append(l)
    l, = ax3.plot(Freq, Fr_Resp_2_Phase)
    l1.append(l)

    ax3.legend(handles=l1, 
               labels=label,
               loc="lower left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
               )
    
    if save_path != None:
        plt.savefig(save_path)




