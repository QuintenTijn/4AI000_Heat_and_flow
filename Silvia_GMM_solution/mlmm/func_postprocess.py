#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:32:41 2021

@author: s20180384
"""

import numpy as np
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colors
from matplotlib import cm
import itertools
from scipy.stats import entropy
from scipy.interpolate import interpn

kB=1.38064852e-23;
conv_v=1.0e2; #convert [An/ps] to [m/s]

def ac_diff(ACM1: np.array,ACM2: np.array) -> np.array:
    """Function that computes the difference between ACs obtained from different approach """
    ac_diff_matrix = []
    for (ac1,ac2) in zip(ACM1,ACM2):
        ac_diff_matrix.append(((ac1-ac2)/ac1)*100)
    return ac_diff_matrix

def liao_transform(vel: np.array,mass: float,direction: str) -> (np.array,float,float,float,float):
    """Function that implements Liao transfer function on MD results"""
    v_mean_in=np.mean(((np.linalg.norm(vel[:,0:3],axis=1).reshape(-1,1))*conv_v)**2)
    v_mean_out=np.mean(((np.linalg.norm(vel[:,3:6],axis=1).reshape(-1,1))*conv_v)**2)
    T_in=v_mean_in*(mass*0.001/6.023e23)/4/kB
    T_out=v_mean_out*(mass*0.001/6.023e23)/4/kB
    theta_in=kB*T_in/(mass*0.001/6.023e23)
    theta_out=kB*T_out/(mass*0.001/6.023e23)
    v_x_in=np.hstack((vel[:,0],-vel[:,0])).reshape(-1,1)
    v_x_out=np.hstack((vel[:,3],-vel[:,3])).reshape(-1,1)
    v_y_in=np.hstack((vel[:,1],-vel[:,1])).reshape(-1,1)
    v_y_out=np.hstack((vel[:,4],-vel[:,4])).reshape(-1,1)
    v_z_in=np.hstack((vel[:,2],-vel[:,2])).reshape(-1,1)
    v_z_out=np.hstack((vel[:,5],-vel[:,5])).reshape(-1,1)
    if direction=='y':
        v_y_in_TF=((np.sqrt(2*theta_in)*erfinv(1.0-2.0*np.exp(-1.0*((v_y_in*conv_v)**2)/(2*theta_in))))/conv_v).reshape(-1,1)
        v_y_out_TF=((np.sqrt(2*theta_out)*erfinv(1.0-2.0*np.exp(-1.0*((v_y_out*conv_v)**2)/(2*theta_out))))/conv_v).reshape(-1,1)
        v_TR=np.concatenate((v_x_in,v_y_in_TF,v_z_in,v_x_out,v_y_out_TF,v_z_out),axis=1)
        return v_TR,T_in,T_out,theta_in,theta_out
    if direction=='z':
        v_z_in_TF=((np.sqrt(2*theta_in)*erfinv(1.0-2.0*np.exp(-1.0*((v_z_in*conv_v)**2)/(2*theta_in))))/conv_v).reshape(-1,1)
        v_z_out_TF=((np.sqrt(2*theta_out)*erfinv(1.0-2.0*np.exp(-1.0*((v_z_out*conv_v)**2)/(2*theta_out))))/conv_v).reshape(-1,1)
        v_TR=np.concatenate((v_x_in,v_y_in,v_z_in_TF,v_x_out,v_y_out,v_z_out_TF),axis=1)
        return v_TR,T_in,T_out,theta_in,theta_out
#---------------------------------------------------------------------------------------------------
def liao_R_transform(vel: np.array,theta_in: float,theta_out: float,direction: str) -> np.array:
    """Function that implements Liao R-transfer function on results"""
    if direction=='y':
        v_y_in_RTF=(np.sqrt(-2.0*theta_in*np.log(0.5-0.5*erf((vel[:,1]*conv_v)/(np.sqrt(2*theta_in))))))/conv_v
        v_y_out_RTF=(np.sqrt(-2.0*theta_out*np.log(0.5-0.5*erf((vel[:,4]*conv_v)/(np.sqrt(2*theta_out))))))/conv_v
        vel2=np.copy(vel)
        vel2[:,1]=v_y_in_RTF
        vel2[:,4]=v_y_out_RTF
        return vel2
    if direction=='z':
        v_z_in_RTF=(np.sqrt(-2.0*theta_in*np.log(0.5-0.5*erf((vel[:,2]*conv_v)/(np.sqrt(2*theta_in))))))/conv_v
        v_z_out_RTF=(np.sqrt(-2.0*theta_out*np.log(0.5-0.5*erf((vel[:,5]*conv_v)/(np.sqrt(2*theta_out))))))/conv_v
        vel2=np.copy(vel)
        vel2[:,2]=v_z_in_RTF
        vel2[:,-1]=v_z_out_RTF
        return vel2
#######################################################################################3
def compute_energy_tot(gas_name: str,vel_tr: np.array,vel_rot: np.array) -> (np.array,np.array):
    """Function that computes total incoming and outgoing energy for a diatomic gas from its TR and ROT velocities"""
    conv_v=1.0e2 #convert [An/ps] to [m/s]
    conv_omega=1.0e12
    av_num=6.022e23
    if gas_name=='H2':
        mass = 2.0158 * 0.001 / av_num
        l_b=0.741e-10
        I=(mass/4)*l_b**2
    if gas_name=='N2':
        mass = 28.02 * 0.001 / av_num
        l_b=1.097e-10
        I = (mass / 4) * l_b ** 2
    vel_used = np.copy(vel_tr)
    vel_tr_SI=vel_used*conv_v
    vel_rot_used = np.copy(vel_rot)
    omega_SI=vel_rot_used*conv_omega
    tr_energy_in=0.5*mass*(np.linalg.norm(vel_tr_SI[:,0:3],axis=1))**2
    tr_energy_out=0.5*mass*(np.linalg.norm(vel_tr_SI[:,3:6],axis=1))**2
    rot_energy_in=0.5*I*(omega_SI[:,0]**2+omega_SI[:,1]**2)
    rot_energy_out=0.5*I*(omega_SI[:,2]**2+omega_SI[:,-1]**2)
    tot_energy_in=tr_energy_in+rot_energy_in
    tot_energy_out=tr_energy_out+rot_energy_out    
    return tot_energy_in, tot_energy_out
##########################################################################################################################
def compute_diff_energy_diatomic(gas_name: str,vel_tr: np.array,vel_rot: np.array) -> (np.array,np.array,np.array):
    """Function that computes TR,ROT, and TOT incoming and outgoing energy for a diatomic gas from its TR and ROT velocities"""
    conv_v = 1.0e2 #convert [An/ps] to [m/s]
    conv_omega = 1.0e12
    av_num = 6.022e23
    if gas_name == 'H2':
        mass = 2.0158 * 0.001 / av_num
        l_b=0.741e-10
        I=(mass/4)*l_b**2
    if gas_name == 'N2':
        mass = 28.02 * 0.001 / av_num
        l_b = 1.097e-10
        I = (mass / 4) * l_b ** 2
    vel_used = np.copy(vel_tr)
    vel_tr_SI = vel_used*conv_v
    vel_rot_used = np.copy(vel_rot)
    omega_SI = vel_rot_used*conv_omega
    tr_energy_in = 0.5*mass*(np.linalg.norm(vel_tr_SI[:,0:3],axis=1))**2
    tr_energy_out = 0.5*mass*(np.linalg.norm(vel_tr_SI[:,3:6],axis=1))**2
    rot_energy_in = 0.5*I*(omega_SI[:,0]**2+omega_SI[:,1]**2)
    rot_energy_out = 0.5*I*(omega_SI[:,2]**2+omega_SI[:,-1]**2)
    tot_energy_in = tr_energy_in+rot_energy_in
    tot_energy_out = tr_energy_out+rot_energy_out
    tr_energy = np.concatenate((tr_energy_in.reshape(-1,1),tr_energy_out.reshape(-1,1)),axis=1)
    rot_energy = np.concatenate((rot_energy_in.reshape(-1,1),rot_energy_out.reshape(-1,1)), axis=1)
    tot_energy = np.concatenate((tot_energy_in.reshape(-1,1), tot_energy_out.reshape(-1,1)), axis=1)
    return tr_energy, rot_energy, tot_energy
    
        
    
###################################################################################################################
def compute_AC_correlation_method(vel: np.array,gas_type: str,direction: str) -> float:
    """Function that computes Different ACs using correlation method for monoatomic and diatomic gases"""
    mono=['Ar','He']
    MAC_x=1-np.polyfit(vel[:,0],vel[:,3],1)[0]
    EAC_x=1-np.polyfit((vel[:,0])**2,(vel[:,3])**2,1)[0]
    EAC_y=1-np.polyfit((vel[:,1])**2,(vel[:,4])**2,1)[0]
    EAC_z=1-np.polyfit((vel[:,2])**2,(vel[:,5])**2,1)[0]
    EAC_tr=1-np.polyfit(((vel[:,0])**2+(vel[:,1])**2+(vel[:,2])**2),((vel[:,3])**2+(vel[:,4])**2+(vel[:,5])**2),1)[0]
    if gas_type in mono:
        if direction == 'y':
            MAC_y = 1 - np.polyfit(np.abs(vel[:, 1]), np.abs(vel[:, 4]), 1)[0]
            MAC_z = 1 - np.polyfit(vel[:, 2], vel[:, 5], 1)[0]
        if direction=='z':
            MAC_y = 1 - np.polyfit(vel[:, 1], vel[:, 4], 1)[0]
            MAC_z = 1 - np.polyfit(np.abs(vel[:, 2]), np.abs(vel[:, 5]), 1)[0]
        return MAC_x,MAC_y,MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr

    if gas_type=='H2':
        mg=2*1.0079*0.001/(6.022e23) #[kg]
        mu=mg/4
        b_l=0.741e-10 #[m]
        I=mu*(b_l**2)
        conv_omega=1.0e12 #convert [1/ps] to [1/s]
        conv_J_2_eV=6.24e18
        vel_tr_SI=vel[:,0:6]*conv_v
        omega_SI=vel[:,6:10]*conv_omega
        tr_energy_in=0.5*mg*(np.linalg.norm(vel_tr_SI[:,0:3],axis=1))**2
        tr_energy_out=0.5*mg*(np.linalg.norm(vel_tr_SI[:,3:6],axis=1))**2
        rot_energy_in=0.5*I*(omega_SI[:,0]**2+omega_SI[:,1]**2)
        rot_energy_out=0.5*I*(omega_SI[:,2]**2+omega_SI[:,-1]**2)
        tot_energy_in=tr_energy_in+rot_energy_in
        tot_energy_out=tr_energy_out+rot_energy_out
        EAC_tr_energy=1-np.polyfit(tr_energy_in[:]*conv_J_2_eV,tr_energy_out[:]*conv_J_2_eV,1)[0]
        EAC_rot=1-np.polyfit(((vel[:,6])**2+(vel[:,7])**2),((vel[:,8])**2+(vel[:,9])**2),1)[0]
        EAC_rot_energy=1-np.polyfit(rot_energy_in[:]*conv_J_2_eV,rot_energy_out[:]*conv_J_2_eV,1)[0]
        EAC_tot_energy=1-np.polyfit(tot_energy_in[:]*conv_J_2_eV,tot_energy_out[:]*conv_J_2_eV,1)[0]
        if direction=='y':    
            MAC_y=1-np.polyfit(np.abs(vel[:,1]),np.abs(vel[:,4]),1)[0]
            MAC_z=1-np.polyfit(vel[:,2],vel[:,5],1)[0]    
        if direction=='z':
            MAC_y=1-np.polyfit(np.abs(vel[:,1]),np.abs(vel[:,4]),1)[0]
            MAC_z=1-np.polyfit(vel[:,2],vel[:,5],1)[0]
        return MAC_x,MAC_y,MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr,EAC_tr_energy,EAC_rot,EAC_rot_energy,EAC_tot_energy

    if gas_type=='N2':
        mg=2*14.006*0.001/(6.022e23) #[kg]
        mu=mg/4
        b_l=1.097e-10 #[m]
        I=mu*(b_l**2)
        conv_omega=1.0e12 #convert [1/ps] to [1/s]
        conv_J_2_eV=6.24e18
        vel_tr_SI=vel[:,0:6]*conv_v
        omega_SI=vel[:,6:]*conv_omega
        tr_energy_in=0.5*mg*(np.linalg.norm(vel_tr_SI[:,0:3],axis=1))**2
        tr_energy_out=0.5*mg*(np.linalg.norm(vel_tr_SI[:,3:6],axis=1))**2
        rot_energy_in=0.5*I*(omega_SI[:,0]**2+omega_SI[:,1]**2)
        rot_energy_out=0.5*I*(omega_SI[:,2]**2+omega_SI[:,-1]**2)
        tot_energy_in=tr_energy_in+rot_energy_in
        tot_energy_out=tr_energy_out+rot_energy_out
        EAC_tr_energy=1-np.polyfit(tr_energy_in[:]*conv_J_2_eV,tr_energy_out[:]*conv_J_2_eV,1)[0]
        EAC_rot=1-np.polyfit(((vel[:,6])**2+(vel[:,7])**2),((vel[:,8])**2+(vel[:,9])**2),1)[0]
        EAC_rot_energy=1-np.polyfit(rot_energy_in[:]*conv_J_2_eV,rot_energy_out[:]*conv_J_2_eV,1)[0]
        EAC_tot_energy=1-np.polyfit(tot_energy_in[:]*conv_J_2_eV,tot_energy_out[:]*conv_J_2_eV,1)[0]
        if direction=='y':
            MAC_y=1-np.polyfit(np.abs(vel[:,1]),np.abs(vel[:,4]),1)[0]
            MAC_z=1-np.polyfit(vel[:,2],vel[:,5],1)[0]
        if direction=='z':
            MAC_y=1-np.polyfit(np.abs(vel[:,1]),np.abs(vel[:,4]),1)[0]
            MAC_z=1-np.polyfit(vel[:,2],vel[:,5],1)[0]
        return MAC_x,MAC_y,MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr,EAC_tr_energy,EAC_rot,EAC_rot_energy,EAC_tot_energy    
##########################################################################################################################################        
def compute_AC_correlation_method_12D(vel,gas_type,direction):
    MAC_x=1-np.polyfit(vel[:,0],vel[:,3],1)[0]
    EAC_x=1-np.polyfit((vel[:,0])**2,(vel[:,3])**2,1)[0]
    EAC_y=1-np.polyfit((vel[:,1])**2,(vel[:,4])**2,1)[0]
    EAC_z=1-np.polyfit((vel[:,2])**2,(vel[:,5])**2,1)[0]
    EAC_tr=1-np.polyfit(((vel[:,0])**2+(vel[:,1])**2+(vel[:,2])**2),((vel[:,3])**2+(vel[:,4])**2+(vel[:,5])**2),1)[0]


    if gas_type=='H2':
        mg=2*1.0079*0.001/(6.022e23) #[kg]
        mu=mg/4
        b_l=0.741e-10 #[m]
        I=mu*(b_l**2)
        conv_omega=1.0e12 #convert [1/ps] to [1/s]
        conv_J_2_eV=6.24e18
        vel_tr_SI=vel[:,0:6]*conv_v
        omega_SI=vel[:,6:10]*conv_omega
        tr_energy_in=0.5*mg*(np.linalg.norm(vel_tr_SI[:,0:3],axis=1))**2
        tr_energy_out=0.5*mg*(np.linalg.norm(vel_tr_SI[:,3:6],axis=1))**2
        rot_energy_in=0.5*I*(omega_SI[:,0]**2+omega_SI[:,1]**2)
        rot_energy_out=0.5*I*(omega_SI[:,2]**2+omega_SI[:,-1]**2)
        tot_energy_in=tr_energy_in+rot_energy_in
        tot_energy_out=tr_energy_out+rot_energy_out
        EAC_tr_energy=1-np.polyfit(tr_energy_in[:]*conv_J_2_eV,tr_energy_out[:]*conv_J_2_eV,1)[0]
        EAC_rot=1-np.polyfit(((vel[:,6])**2+(vel[:,7])**2),((vel[:,8])**2+(vel[:,9])**2),1)[0]
        EAC_rot_energy=1-np.polyfit(rot_energy_in[:]*conv_J_2_eV,rot_energy_out[:]*conv_J_2_eV,1)[0]
        EAC_tot_energy=1-np.polyfit(tot_energy_in[:]*conv_J_2_eV,tot_energy_out[:]*conv_J_2_eV,1)[0]
        EAC_tot_12D=1-np.polyfit(vel[:,10]*conv_J_2_eV,vel[:,11]*conv_J_2_eV,1)[0]
        if direction=='y':    
            MAC_y=1-np.polyfit(np.abs(vel[:,1]),np.abs(vel[:,4]),1)[0]
            MAC_z=1-np.polyfit(vel[:,2],vel[:,5],1)[0]    
        if direction=='z':
            MAC_y=1-np.polyfit(np.abs(vel[:,1]),np.abs(vel[:,4]),1)[0]
            MAC_z=1-np.polyfit(vel[:,2],vel[:,5],1)[0]
        return MAC_x,MAC_y,MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr,EAC_tr_energy,EAC_rot,EAC_rot_energy,EAC_tot_energy,EAC_tot_12D

    if gas_type=='N2':
        mg=2*14.006*0.001/(6.022e23) #[kg]
        mu=mg/4
        b_l=1.097e-10 #[m]
        I=mu*(b_l**2)
        conv_omega=1.0e12 #convert [1/ps] to [1/s]
        conv_J_2_eV=6.24e18
        vel_tr_SI=vel[:,0:6]*conv_v
        omega_SI=vel[:,6:]*conv_omega
        tr_energy_in=0.5*mg*(np.linalg.norm(vel_tr_SI[:,0:3],axis=1))**2
        tr_energy_out=0.5*mg*(np.linalg.norm(vel_tr_SI[:,3:6],axis=1))**2
        rot_energy_in=0.5*I*(omega_SI[:,0]**2+omega_SI[:,1]**2)
        rot_energy_out=0.5*I*(omega_SI[:,2]**2+omega_SI[:,-1]**2)
        tot_energy_in=tr_energy_in+rot_energy_in
        tot_energy_out=tr_energy_out+rot_energy_out
        EAC_tr_energy=1-np.polyfit(tr_energy_in[:]*conv_J_2_eV,tr_energy_out[:]*conv_J_2_eV,1)[0]
        EAC_rot=1-np.polyfit(((vel[:,6])**2+(vel[:,7])**2),((vel[:,8])**2+(vel[:,9])**2),1)[0]
        EAC_rot_energy=1-np.polyfit(rot_energy_in[:]*conv_J_2_eV,rot_energy_out[:]*conv_J_2_eV,1)[0]
        EAC_tot_energy=1-np.polyfit(tot_energy_in[:]*conv_J_2_eV,tot_energy_out[:]*conv_J_2_eV,1)[0]
        EAC_tot_12D=1-np.polyfit(vel[:,10]*conv_J_2_eV,vel[:,11]*conv_J_2_eV,1)[0]
        if direction=='y':
            MAC_y=1-np.polyfit(np.abs(vel[:,1]),np.abs(vel[:,4]),1)[0]
            MAC_z=1-np.polyfit(vel[:,2],vel[:,5],1)[0]
        if direction=='z':
            MAC_y=1-np.polyfit(np.abs(vel[:,1]),np.abs(vel[:,4]),1)[0]
            MAC_z=1-np.polyfit(vel[:,2],vel[:,5],1)[0]
        return MAC_x,MAC_y,MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr,EAC_tr_energy,EAC_rot,EAC_rot_energy,EAC_tot_energy,EAC_tot_12D 
####################################################################################################################
def plot_normal_velocity_hist(vel: np.array,direction: str,data_type: str,path_file: str):
    """Function to plot incoming and outgoing normal velocity components"""
    if direction=='y':
        plt.figure()
        plt.subplot(121)
        plt.hist(vel[:,1])
        plt.title(data_type+direction+'_in')
        plt.subplot(122)
        plt.hist(vel[:,4])
        plt.title(data_type+direction+'_out')
        plt.savefig(path_file+'/'+data_type+'.png')
    if direction=='z':
        plt.figure()
        plt.subplot(121)
        plt.hist(vel[:,2])
        plt.title(data_type+direction+'_in')
        plt.subplot(122)
        plt.hist(vel[:,5])
        plt.title(data_type+direction+'_in')
        plt.savefig(path_file+'/'+data_type+'.png')
#####################################################################################################################################################
def velocity_CLL(vel_matrix: np.array,Ts: float,gas_name: str) -> np.array:
    """Function to compute CLL velocity components for monoatomic and diatomic gases"""
    kB=1.38064852e-23
    conv_v=1.0e2 #convert [An/ps] to [m/s]
    conv_omega=1.0e12
    av_num=6.022e23
    if gas_name=='He':
        mass=4.002*0.001/av_num
    if gas_name=='Ar':
        mass = 39.948 * 0.001 / av_num
    if gas_name=='H2':
        mass = 2.0158 * 0.001 / av_num
        l_b=0.741e-10
        I=(mass/4)*l_b**2
    if gas_name=='N2':
        mass = 28.02 * 0.001 / av_num
        l_b=1.097e-10
        I = (mass / 4) * l_b ** 2
    v_mp = np.sqrt(2 * kB * Ts / mass)
    MAC_x, MAC_y, MAC_z, EAC_x, EAC_y, EAC_z, EAC_tr = compute_AC_correlation_method(vel_matrix,'Ar', 'y')
    alpha_t1_cll = MAC_x * (2 - MAC_x)
    alpha_t2_cll = MAC_z * (2 - MAC_z)
    vel_t1 = (vel_matrix[:, 0] * conv_v / v_mp) * np.sqrt(1 - alpha_t1_cll)
    vel_x_cll = np.zeros((vel_matrix.shape[0], 1))
    for i in range(vel_matrix.shape[0]):
        rand_num = np.random.rand(2)
        r1 = np.sqrt(-alpha_t1_cll * np.log(rand_num[0]))
        theta_1 = 2 * np.pi * rand_num[1]
        vel_x_cll[i] = v_mp * (vel_t1[i] + (r1 * np.cos(theta_1)))
    #########################################################################################3
    vel_t2 = (vel_matrix[:, 2] * conv_v / v_mp) * np.sqrt(1 - alpha_t2_cll)
    vel_z_cll = np.zeros((vel_matrix.shape[0], 1))
    for i in range(vel_matrix.shape[0]):
        rand_num = np.random.rand(2)
        r1 = np.sqrt(-alpha_t2_cll * np.log(rand_num[0]))
        theta_1 = 2 * np.pi * rand_num[1]
        vel_z_cll[i] = v_mp * (vel_t2[i] + (r1 * np.sin(theta_1)))
    ################################################################################################
    vel_n = (vel_matrix[:, 1] * conv_v / v_mp) * np.sqrt(1 - EAC_y)
    vel_y_cll = np.zeros((vel_matrix.shape[0], 1))
    for i in range(vel_matrix.shape[0]):
        rand_num = np.random.rand(2)
        r1 = np.sqrt(-EAC_y * np.log(rand_num[0]))
        theta_1 = 2 * np.pi * rand_num[1]
        vel_y_cll[i] = v_mp * np.sqrt(r1 ** 2 + vel_n[i] ** 2 + 2 * r1 * vel_n[i] * np.cos(theta_1))
    if vel_matrix.shape[1]==6:
        vel_cll=np.concatenate((vel_x_cll,vel_y_cll,vel_z_cll),axis=1)
        vel_cll_tot=np.concatenate((vel_matrix[:,0:3],vel_cll/conv_v),axis=1)
        return vel_cll_tot
    if vel_matrix.shape[1]==10:
        omega_mp = np.sqrt(2 * kB * Ts / I)
        MAC_x_MD, MAC_y_MD, MAC_z_MD, EAC_x_MD, EAC_y_MD, EAC_z_MD, EAC_tr_MD, EAC_tr_energy_MD,\
        EAC_rot_MD, EAC_rot_energy_MD,\
        EAC_tot_energy_MD = compute_AC_correlation_method(vel_matrix, gas_name, 'y')
        omega_1 = (vel_matrix[:, 6] * conv_omega / omega_mp) * np.sqrt(1 - EAC_rot_MD)
        omega_2 = (vel_matrix[:, 7] * conv_omega / omega_mp) * np.sqrt(1 - EAC_rot_MD)
        omega_1_cll = np.zeros((vel_matrix.shape[0], 1))
        omega_2_cll = np.zeros((vel_matrix.shape[0], 1))
        for i in range(vel_matrix.shape[0]):
            rand_num = np.random.rand(2)
            r1 = np.sqrt(-EAC_rot_MD * np.log(rand_num[0]))
            theta_1 = 2 * np.pi * rand_num[1]
            omega_1_cll[i] = omega_mp * (omega_1[i] + (r1 * np.sin(theta_1)))
            #############################################################################
        for i in range(vel_matrix.shape[0]):
            rand_num = np.random.rand(2)
            r1 = np.sqrt(-EAC_rot_MD * np.log(rand_num[0]))
            theta_1 = 2 * np.pi * rand_num[1]
            omega_2_cll[i] = omega_mp * (omega_2[i] + (r1 * np.sin(theta_1)))
        vel_cll = np.concatenate((vel_x_cll/conv_v, vel_y_cll/conv_v, vel_z_cll/conv_v,\
                                  omega_1_cll/conv_omega,omega_2_cll/conv_omega), axis=1)
        vel_cll_tot=np.concatenate((vel_matrix[:,0:3],vel_cll[:,0:3],vel_matrix[:,6:8],vel_cll[:,3:]),axis=1)
        return vel_cll_tot
#############################################################################################
def lines_heat_map(v_in: np.array,v_out: np.array,yes_abs: bool = False ) -> (np.array,np.array,np.array):
    """Function to compute refelcite, diffuse and best least square fit into incoming-outgoing velocities"""
    if yes_abs:
        v_in = np.abs(v_in)
        v_out = np.abs(v_out)
    pc = np.polyfit(v_in,v_out, 1)
    coff1 = pc[0]
    coff2 = pc[1]
    v_min = np.min(v_in)
    v_max = np.max(v_in)
    ref_line = 2*np.linspace(2*v_min,2*v_max,100)
    dif_line = np.mean(v_out) * np.ones((len(ref_line)))
    y_fit = coff1 * ref_line+coff2
    return ref_line, dif_line, y_fit
######################################################################################################################
def density_scatter( x: np.array , y: np.array, ax = None, sort = True, bins = 20, **kwargs ):
    """
    Scatter plot colored by 2d histogram
    """
    #import numpy as np
    #import matplotlib.pyplot as plt
    #from matplotlib import cm
    #from matplotlib.colors import Normalize
    #from scipy.interpolate import interpn
    x_axis_min = np.min(x)
    x_axis_max = np.max(x)
    y_axis_min = np.min(y)
    y_axis_max = np.max(y)
    if x_axis_min >= 0:
        x_axis_min = 0
        x_axis_max = max(x_axis_max,y_axis_max)
        y_axis_max = x_axis_max
    if y_axis_min >= 0:
        y_axis_min = 0
    if ax is None:
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    ref_line, dif_line, y_fit = lines_heat_map(x, y)

    ax.scatter( x, y, c=z, **kwargs )
    ax.plot(ref_line,y_fit,color='red',linewidth=2)
    ax.plot(ref_line,ref_line,linestyle='--',color='black',linewidth=2)
    ax.plot(ref_line,dif_line,linestyle='--',color='black',linewidth=2)
    ax.axis('equal')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(x_axis_min, x_axis_max)
    ax.set_ylim(y_axis_min, y_axis_max)
    #ax.scatter( x, y, c=z, cmap='bwr' )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    #cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    #cbar.ax.set_ylabel('Density')

    return ax
#####################################################################################################
def conv_hist_to_bin(vel: np.array, num_bin: float=30) -> (np.array,np.array):
    """Function to derive PDF from histogram graph"""
    num_cul = vel.shape[1]
    hist_values_all = np.zeros((num_bin,num_cul))
    bin_middle_all = np.zeros((num_bin,num_cul))
    bin_middle = np.zeros((num_bin,1))
    for i in range(num_cul):
        v_min = vel[:, i].min()
        v_max = vel[:, i].max()
        hist_values, bin_edges = np.histogram(vel[:, i], bins=num_bin,range=(v_min, v_max), density=True)
        hist_values_all[:, i] = hist_values
        for j in range(0, len(bin_edges) -1):
            bin_value = (bin_edges[j] + bin_edges[j + 1]) / 2
            bin_middle[j] = bin_value
        bin_middle_all[:, i] = bin_middle[:, 0]
    return hist_values_all,bin_middle_all
#####################################################################################################
#def plot_pdf_tr_velocities_hist(vel,num_bins,data_type,path_file):
def plot_pdf_tr_velocities_hist(*args):
    '''
    Can be used to compare the PDFs of incoming and outgoing velocities from MD simulation. 
    
    'MD'
    (arg[0]='MD', arg[1]= N_bin, arg[2] = MD test velocity matrix, arg[3] = path_for_saving_fig
   ----------------------------------------------------------------------------------------------------
   'MD-diatomic'
   arg[0]='MD', arg[1]= N_bin, arg[2] = MD test velocity matrix, arg[3] = path_for_saving_fig
   -----------------------------------------------------------------------------------------------------
     Can be used to compare velocity correlation and PDFs of velocities obtainded from MD and ML 6D :
         
    'ML':
      (arg[0]='MD', arg[1]= N_bin, arg[2]= MD test velocity matrix, arg[3]=ML velocity matrix, arg[4]= path_for_saving_fig
      arg[5]= NG
     -------------------------------------------------------------------------------------------------
      Can be used to compare velocity correlation and PDFs of velocities obtainded from MD, CLL, ML 6D :
          
    'ML-CLL'  for 6D data
      ('MD'=arg[0], N_bin=arg[1], MD test velocity matrix=arg[2],ML velocity matrix=arg[3],CLL velocity matrix=arg[4], path_for_saving_fig=arg[5]
      NG=arg[6]
      ----------------------------------------------------------
      ML-multi: for 6D data
          args[0]=ML-multi, args[1]=Nbin, args[2]=MD velocity, args[3]=ML velocity matrix, args[4]=path
          args[5]=NG, args[6]=Data set name
     ---------------------------------------------------------------------------------------------
     'ML-10D_rot_En'
     args[0]=ML-10D, args[1]=gas name, args[2]=Nbin, args[3]=MD velocity matrix, args[4]=ML velocity matrix
          args[5]=path, args[6]=NG
    -----------------------------------------------------------------------------------------------
    'ML-10D_rot_vel'
    args[0]=ML-10D_rot_vel, args[1]=gas name, args[2]=Nbin, args[3]=MD velocity matrix, args[4]=ML velocity matrix
          args[5]=path, args[6]=NG


     
    '''
    if args[0]=='MD':
        hist_values_all = np.zeros((args[1], 6)) # arg[1] num bins
        bin_middle_all = np.zeros((args[1], 6))
        for i in range(6):
            v_min = args[2][:, i].min()
            v_max = args[2][:, i].max()
            hist_values, bin_edges = np.histogram(args[2][:, i], bins=args[1],range=(v_min, v_max), density=True)
            hist_values_all[:, i] = hist_values
            bin_middle = np.zeros((args[1], 1))
            for j in range(0, len(bin_edges) - 1):
                bin_value = (bin_edges[j] + bin_edges[j + 1]) / 2
                bin_middle[j] = bin_value
            bin_middle_all[:, i] = bin_middle[:, 0]
        fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
        axs[0].plot(bin_middle_all[:, 0], hist_values_all[:, 0], label='$Vx_{in}$')
        axs[0].plot(bin_middle_all[:, 3], hist_values_all[:, 3], label='$Vx_{out}$')
        legend = axs[0].legend()
        legend.get_frame().set_edgecolor('white')
        axs[1].plot(bin_middle_all[:, 1], hist_values_all[:, 1], label='$Vy_{in}$')
        axs[1].plot(bin_middle_all[:, 4], hist_values_all[:, 4], label='$Vy_{out}$')
        legend = axs[1].legend()
        legend.get_frame().set_edgecolor('white')
        axs[2].plot(bin_middle_all[:, 2], hist_values_all[:, 2], label='$Vz_{in}$')
        axs[2].plot(bin_middle_all[:, 5], hist_values_all[:, 5], label='$Vz_{out}$')
        legend = axs[2].legend()
        legend.get_frame().set_edgecolor('white')
        fig.suptitle(args[0] + ' results')
        plt.savefig(args[3] + '/Vin_' +'Vout_'+ args[0] + '.png')
    ###############################################################################################
    if args[0]=='MD-diatomic':
        hist_values_all = np.zeros((args[1], 10))
        bin_middle_all = np.zeros((args[1], 10))
        for i in range(10):
            v_min = args[2][:, i].min()
            v_max = args[2][:, i].max()
            hist_values, bin_edges = np.histogram(args[2][:, i], bins=args[1],range=(v_min, v_max), density=True)
            hist_values_all[:, i] = hist_values
            bin_middle = np.zeros((args[1], 1))
            for j in range(0, len(bin_edges) - 1):
                bin_value = (bin_edges[j] + bin_edges[j + 1]) / 2
                bin_middle[j] = bin_value
            bin_middle_all[:, i] = bin_middle[:, 0]
        fig, axs = plt.subplots(1, 5, figsize=(15, 6), sharey=True)
        axs[0].plot(bin_middle_all[:, 0], hist_values_all[:, 0], label='$Vx_{in}$')
        axs[0].plot(bin_middle_all[:, 3], hist_values_all[:, 3], label='$Vx_{out}$')
        legend = axs[0].legend()
        legend.get_frame().set_edgecolor('white')
        axs[1].plot(bin_middle_all[:, 1], hist_values_all[:, 1], label='$Vy_{in}$')
        axs[1].plot(bin_middle_all[:, 4], hist_values_all[:, 4], label='$Vy_{out}$')
        legend = axs[1].legend()
        legend.get_frame().set_edgecolor('white')
        axs[2].plot(bin_middle_all[:, 2], hist_values_all[:, 2], label='$Vz_{in}$')
        axs[2].plot(bin_middle_all[:, 5], hist_values_all[:, 5], label='$Vz_{out}$')
        legend = axs[2].legend()
        legend.get_frame().set_edgecolor('white')
        axs[3].plot(bin_middle_all[:, 6], hist_values_all[:, 6], label='$W1_{in}$')
        axs[3].plot(bin_middle_all[:, 8], hist_values_all[:, 8], label='$W1_{out}$')
        legend = axs[3].legend()
        legend.get_frame().set_edgecolor('white')
        axs[4].plot(bin_middle_all[:, 7], hist_values_all[:, 7], label='$W1_{in}$')
        axs[4].plot(bin_middle_all[:, 9], hist_values_all[:, 9], label='$W1_{out}$')
        legend = axs[3].legend()
        legend.get_frame().set_edgecolor('white')
        fig.suptitle(args[0] + ' results')
        plt.savefig(args[3] + '/Vin_' +'Vout_'+ args[0] + '.png')
        ##############################################################################
    if args[0]=='ML':

        num_test=args[2].shape[0]
        num_train = args[3].shape[0]
        hist_values_all_MD = np.zeros((args[1], 6))
        bin_middle_all_MD = np.zeros((args[1], 6))
        hist_values_all_ML = np.zeros((args[1], 6))
        bin_middle_all_ML = np.zeros((args[1], 6))
        for i in range(6):
            v_min = args[2][:, i].min()
            v_max = args[2][:, i].max()
            hist_values, bin_edges = np.histogram(args[2][:, i], bins=args[1],range=(v_min, v_max), density=True)
            hist_values_ML, bin_edges_ML = np.histogram(args[3][:, i], bins=args[1],range=(v_min, v_max), density=True)
            hist_values_all_MD[:, i] = hist_values
            hist_values_all_ML[:, i] = hist_values_ML
            bin_middle = np.zeros((args[1], 1))
            bin_middle_ML = np.zeros((args[1], 1))
            for j in range(0, len(bin_edges) - 1):
                bin_value = (bin_edges[j] + bin_edges[j + 1]) / 2
                bin_value_ML = (bin_edges_ML[j] + bin_edges_ML[j + 1]) / 2
                bin_middle[j] = bin_value
                bin_middle_ML[j] = bin_value_ML
            bin_middle_all_MD[:, i] = bin_middle[:, 0]
            bin_middle_all_ML[:, i] = bin_middle_ML[:, 0]
        fig, axs = plt.subplots(3, 3, figsize=(9, 6),tight_layout=True)
        axs[0,0].hist2d(args[2][:, 0], args[2][:, 3], bins=40, norm=colors.LogNorm())
        axs[0,1].hist2d(args[3][:, 0], args[3][:, 3], bins=40, norm=colors.LogNorm())
        axs[0,2].plot(bin_middle_all_MD[:, 3], hist_values_all_MD[:, 3], label='$Vx_{MD}$')
        axs[0,2].plot(bin_middle_all_ML[:, 3], hist_values_all_ML[:, 3], label='$Vx_{ML}$')
        legend = axs[0,2].legend()
        legend.get_frame().set_edgecolor('white')
        axs[1, 0].hist2d(args[2][:, 1], args[2][:, 4], bins=40, norm=colors.LogNorm())
        axs[1, 1].hist2d(args[3][:, 1], args[3][:, 4], bins=40, norm=colors.LogNorm())
        axs[1,2].plot(bin_middle_all_MD[:, 4], hist_values_all_MD[:, 4], label='$Vy_{MD}$')
        axs[1,2].plot(bin_middle_all_ML[:, 4], hist_values_all_ML[:, 4], label='$Vy_{ML}$')
        legend = axs[1,2].legend()
        legend.get_frame().set_edgecolor('white')
        axs[2, 0].hist2d(args[2][:, 2], args[2][:, 5], bins=40, norm=colors.LogNorm())
        axs[2, 1].hist2d(args[3][:, 2], args[3][:, 5], bins=40, norm=colors.LogNorm())
        axs[2,2].plot(bin_middle_all_MD[:, 5], hist_values_all_MD[:, 5], label='$Vz_{MD}$')
        axs[2,2].plot(bin_middle_all_ML[:, 5], hist_values_all_ML[:, 5], label='$Vz_{ML}$')
        legend = axs[2,2].legend()
        legend.get_frame().set_edgecolor('white')

        fig.suptitle('Comparison between MD and ML results NG= '+str(args[5]))
        #else:
        #    fig.suptitle('Comparison between MD and ML results NG= ' + str(args[5]))

        plt.savefig(args[4] + '/' + args[0] +'_Ntrain_'+str(num_train)+'_Ntest_'+str(num_test)+'_NG_'+str(args[5])+'.png')
        ###################################################################################################################################
    if args[0]=='ML-10D_rot_En':
        av_num = 6.022e23
        conv_omega = 1.0e12  # convert [1/ps] to [1/s]
        conv_J_2_eV = 6.24e18
        if args[1] == 'H2':
            mass = 2.0158 * 0.001 / av_num
            l_b = 0.741e-10
            l_b_v = 0.741
            I = (mass / 4) * l_b ** 2
        if args[1] == 'N2':
            mass = 28.02 * 0.001 / av_num
            l_b = 1.097e-10
            l_b_v = 1.097
            I = (mass / 4) * l_b ** 2
        MD_results = np.copy(args[3])
        ML_results = np.copy(args[4])
        omega_SI_MD = MD_results[:, 6:10] * conv_omega
        omega_SI_pred = ML_results[:,6:]*conv_omega

        #omega_SI_pred = args[4][:, 6:10]
        rot_energy_in_MD = (0.5 * I * (omega_SI_MD[:, 0] ** 2 + omega_SI_MD[:, 1] ** 2))*conv_J_2_eV
        rot_energy_out_MD = (0.5 * I * (omega_SI_MD[:, 2] ** 2 + omega_SI_MD[:, 3] ** 2))*conv_J_2_eV
        rot_energy_in_pred = (0.5 * I * (omega_SI_pred[:, 0] ** 2 + omega_SI_pred[:, 1] ** 2))*conv_J_2_eV
        rot_energy_out_pred = (0.5 * I * (omega_SI_pred[:, 2] ** 2 + omega_SI_pred[:, 3] ** 2))*conv_J_2_eV
        MD_results = np.concatenate((MD_results[:,:6],rot_energy_in_MD.reshape((-1,1)),rot_energy_out_MD.reshape((-1,1))),axis=1)
        ML_results = np.concatenate((ML_results[:,:6], rot_energy_in_pred.reshape((-1,1)), rot_energy_out_pred.reshape((-1,1))), axis=1)

        num_test=args[3].shape[0]
        num_train = args[4].shape[0]
        hist_values_all_MD = np.zeros((args[2], 8))
        bin_middle_all_MD = np.zeros((args[2], 8))
        hist_values_all_ML = np.zeros((args[2], 8))
        bin_middle_all_ML = np.zeros((args[2], 8))
        for i in range(8):
            #v_min = args[3][:, i].min()
            #v_max = args[3][:, i].max()
            hist_values, bin_edges = np.histogram(MD_results[:, i], bins=args[2], density=True)
            hist_values_ML, bin_edges_ML = np.histogram(ML_results[:, i], bins=args[2], density=True)
            hist_values_all_MD[:, i] = hist_values
            hist_values_all_ML[:, i] = hist_values_ML
            bin_middle = np.zeros((args[2], 1))
            bin_middle_ML = np.zeros((args[2], 1))
            for j in range(0, len(bin_edges) - 1):
                bin_value = (bin_edges[j] + bin_edges[j + 1]) / 2
                bin_value_ML = (bin_edges_ML[j] + bin_edges_ML[j + 1]) / 2
                bin_middle[j] = bin_value
                bin_middle_ML[j] = bin_value_ML
            bin_middle_all_MD[:, i] = bin_middle[:, 0]
            bin_middle_all_ML[:, i] = bin_middle_ML[:, 0]
        fig, axs = plt.subplots(4, 3, figsize=(9, 6),tight_layout=True)
        axs[0,0].hist2d(MD_results[:, 0], MD_results[:, 3], bins=args[2], norm=colors.LogNorm())
        axs[0,1].hist2d(ML_results[:, 0], ML_results[:, 3], bins=args[2], norm=colors.LogNorm())
        axs[0,2].plot(bin_middle_all_MD[:, 3], hist_values_all_MD[:, 3], label='$Vx_{MD}$')
        axs[0,2].plot(bin_middle_all_ML[:, 3], hist_values_all_ML[:, 3], label='$Vx_{ML}$')
        legend = axs[0,2].legend()
        legend.get_frame().set_edgecolor('white')
        axs[1, 0].hist2d(MD_results[:, 1], MD_results[:, 4], bins=args[2], norm=colors.LogNorm())
        axs[1, 1].hist2d(ML_results[:, 1], ML_results[:, 4], bins=args[2], norm=colors.LogNorm())
        axs[1,2].plot(bin_middle_all_MD[:, 4], hist_values_all_MD[:, 4], label='$Vy_{MD}$')
        axs[1,2].plot(bin_middle_all_ML[:, 4], hist_values_all_ML[:, 4], label='$Vy_{ML}$')
        legend = axs[1,2].legend()
        legend.get_frame().set_edgecolor('white')
        axs[2, 0].hist2d(MD_results[:, 2], MD_results[:, 5], bins=args[2], norm=colors.LogNorm())
        axs[2, 1].hist2d(ML_results[:, 2], ML_results[:, 5], bins=args[2], norm=colors.LogNorm())
        axs[2,2].plot(bin_middle_all_MD[:, 5], hist_values_all_MD[:, 5], label='$Vz_{MD}$')
        axs[2,2].plot(bin_middle_all_ML[:, 5], hist_values_all_ML[:, 5], label='$Vz_{ML}$')
        legend = axs[2,2].legend()
        legend.get_frame().set_edgecolor('white')
        axs[3, 0].hist2d(MD_results[:, 6], MD_results[:, 7], bins=args[2], norm=colors.LogNorm())
        axs[3, 1].hist2d(ML_results[:, 6], ML_results[:, 7], bins=args[2], norm=colors.LogNorm())
        axs[3,2].plot(bin_middle_all_MD[:, 7], hist_values_all_MD[:, 7], label='$Erot_{MD}$')
        axs[3,2].plot(bin_middle_all_ML[:, 7], hist_values_all_ML[:, 7], label='$Erot_{ML}$')
        legend = axs[3,2].legend()
        legend.get_frame().set_edgecolor('white')

        fig.suptitle('Comparison between MD and ML results NG= '+str(args[6]))
        #else:
        #    fig.suptitle('Comparison between MD and ML results NG= ' + str(args[5]))

        plt.savefig(args[5] + '/' + args[0] +'_Ntrain_'+str(num_train)+'_Ntest_'+str(num_test)+'_NG_'+str(args[6])+'.png')
        ###################################################################################################################################
    if args[0] == 'ML-10D_CLL_rot_En':
        conv_J_2_eV = 6.24e18
        MD_results = np.copy(args[3])
        ML_results = np.copy(args[4])
        CLL_results = np.copy(args[5])
        MD_results[:,1], MD_results[:,4] = np.abs(MD_results[:,1]), np.abs(MD_results[:,4]) # in order to have same sign for scatter plot
        ML_results[:, 1], ML_results[:, 4] = np.abs(ML_results[:, 1]), np.abs(ML_results[:, 4])
        CLL_results[:, 1], CLL_results[:, 4] = np.abs(CLL_results[:, 1]), np.abs(CLL_results[:, 4])
        tr_energy_MD,rot_energy_MD,tot_energy_MD = compute_diff_energy_diatomic(args[1],MD_results[:, :6],MD_results[:, 6:10])
        tr_energy_CLL, rot_energy_CLL, tot_energy_CLL = compute_diff_energy_diatomic(args[1], CLL_results[:, :6],CLL_results[:, 6:])
        tr_energy_pred, rot_energy_pred, tot_energy_pred = compute_diff_energy_diatomic(args[1], ML_results[:, :6], ML_results[:, 6:])
        MD_results = np.concatenate((MD_results[:, :6],rot_energy_MD*conv_J_2_eV),axis=1)
        ML_results = np.concatenate((ML_results[:, :6], rot_energy_pred*conv_J_2_eV), axis=1)
        CLL_results = np.concatenate((CLL_results[:, :6], rot_energy_CLL*conv_J_2_eV), axis=1)
        num_test=args[3].shape[0]
        num_train = args[4].shape[0]
        hist_values_all_MD,bin_middle_all_MD = conv_hist_to_bin(MD_results)
        hist_values_all_CLL, bin_middle_all_CLL = conv_hist_to_bin(CLL_results)
        hist_values_all_ML, bin_middle_all_ML = conv_hist_to_bin(ML_results)
        fig, axs = plt.subplots(4, 4, figsize=(12, 10),tight_layout=True)
        density_scatter(MD_results[::4, 0], MD_results[::4, 3], ax= axs[0,0])
        density_scatter(ML_results[::4, 0], ML_results[::4, 3], ax=axs[0, 1])
        density_scatter(CLL_results[::4, 0], CLL_results[::4, 3], ax=axs[0, 2])
        axs[0,3].plot(bin_middle_all_MD[:, 3], hist_values_all_MD[:, 3], label='$Vx_{MD}$')
        axs[0,3].plot(bin_middle_all_ML[:, 3], hist_values_all_ML[:, 3], label='$Vx_{ML}$')
        axs[0,3].plot(bin_middle_all_CLL[:, 3], hist_values_all_CLL[:, 3], label='$Vx_{CLL}$')
        legend = axs[0,3].legend()
        legend.get_frame().set_edgecolor('white')
        density_scatter(MD_results[::4, 1], MD_results[::4, 4], ax=axs[1, 0])
        density_scatter(ML_results[::4, 1], ML_results[::4, 4], ax=axs[1, 1])
        density_scatter(CLL_results[::4, 1], CLL_results[::4, 4], ax=axs[1, 2])
        axs[1,3].plot(bin_middle_all_MD[:, 4], hist_values_all_MD[:, 4], label='$Vy_{MD}$')
        axs[1,3].plot(bin_middle_all_ML[:, 4], hist_values_all_ML[:, 4], label='$Vy_{ML}$')
        axs[1, 3].plot(bin_middle_all_CLL[:, 4], hist_values_all_CLL[:, 4], label='$Vy_{CLL}$')
        legend = axs[1,3].legend()
        legend.get_frame().set_edgecolor('white')
        density_scatter(MD_results[::4, 2], MD_results[::4, 5], ax=axs[2, 0])
        density_scatter(ML_results[::4, 2], ML_results[::4, 5], ax=axs[2, 1])
        density_scatter(CLL_results[::4, 2], CLL_results[::4, 5], ax=axs[2, 2])
        axs[2,3].plot(bin_middle_all_MD[:, 5], hist_values_all_MD[:, 5], label='$Vz_{MD}$')
        axs[2,3].plot(bin_middle_all_ML[:, 5], hist_values_all_ML[:, 5], label='$Vz_{ML}$')
        axs[2, 3].plot(bin_middle_all_CLL[:, 5], hist_values_all_CLL[:, 5], label='$Vz_{CLL}$')
        legend = axs[2,3].legend()
        legend.get_frame().set_edgecolor('white')
        density_scatter(MD_results[::4, 6], MD_results[::4, 7], ax=axs[3, 0])
        density_scatter(ML_results[::4, 6], ML_results[::4, 7], ax=axs[3, 1])
        density_scatter(CLL_results[::4, 6], CLL_results[::4, 7], ax=axs[3, 2])
        axs[3,3].plot(bin_middle_all_MD[:, 7], hist_values_all_MD[:, 7], label='$Erot_{MD}$')
        axs[3,3].plot(bin_middle_all_ML[:, 7], hist_values_all_ML[:, 7], label='$Erot_{ML}$')
        axs[3, 3].plot(bin_middle_all_CLL[:, 7], hist_values_all_CLL[:, 7], label='$Erot_{CLL}$')
        legend = axs[3,3].legend()
        legend.get_frame().set_edgecolor('white')

        fig.suptitle('Comparison between MD and ML results NG= '+str(args[7]))
        #else:
        #    fig.suptitle('Comparison between MD and ML results NG= ' + str(args[5]))

        plt.savefig(args[6] + '/' + args[0] +'_Ntrain_'+str(num_train)+'_Ntest_'+str(num_test)+'_NG_'+str(args[7])+'.png',dpi=300)
     #######################################################################################################################################################
    if args[0] == 'ML-10D_CLL_only_energy':
        conv_J_2_eV = 6.24e18
        MD_results = np.copy(args[3])
        ML_results = np.copy(args[4])
        CLL_results = np.copy(args[5])
        MD_results[:, 1], MD_results[:, 4] = np.abs(MD_results[:, 1]), np.abs(MD_results[:, 4])  # in order to have same sign for scatter plot
        ML_results[:, 1], ML_results[:, 4] = np.abs(ML_results[:, 1]), np.abs(ML_results[:, 4])
        CLL_results[:, 1], CLL_results[:, 4] = np.abs(CLL_results[:, 1]), np.abs(CLL_results[:, 4])
        tr_en_MD,rot_en_MD,tot_en_MD = compute_diff_energy_diatomic(args[1],MD_results[:,:6],MD_results[:,6:])
        tr_en_CLL, rot_en_CLL, tot_en_CLL = compute_diff_energy_diatomic(args[1], CLL_results[:, :6], CLL_results[:, 6:])
        tr_en_ML, rot_en_ML, tot_en_ML = compute_diff_energy_diatomic(args[1], ML_results[:, :6], ML_results[:, 6:])
        energy_CLL = (np.concatenate((tr_en_CLL, rot_en_CLL, tot_en_CLL),axis = 1))* conv_J_2_eV
        energy_MD = (np.concatenate((tr_en_MD,rot_en_MD,tot_en_MD), axis=1)) * conv_J_2_eV
        energy_ML = (np.concatenate((tr_en_ML, rot_en_ML, tot_en_ML), axis=1)) * conv_J_2_eV
        num_test = args[3].shape[0]
        num_train = args[4].shape[0]
        hist_values_all_MD,bin_middle_all_MD = conv_hist_to_bin(energy_MD)
        hist_values_all_CLL, bin_middle_all_CLL = conv_hist_to_bin(energy_CLL)
        hist_values_all_ML, bin_middle_all_ML = conv_hist_to_bin(energy_ML)
        fig, axs = plt.subplots(3, 4, figsize=(12, 10), tight_layout=True)
        density_scatter(energy_MD[::4, 0], energy_MD[::4, 1], ax=axs[0, 0])
        density_scatter(energy_ML[::4, 0], energy_ML[::4, 1], ax=axs[0, 1])
        density_scatter(energy_CLL[::4, 0], energy_CLL[::4, 1], ax=axs[0, 2])
        axs[0, 3].plot(bin_middle_all_MD[:, 1], hist_values_all_MD[:, 1], label='$tr_{MD}$')
        axs[0, 3].plot(bin_middle_all_ML[:, 1], hist_values_all_ML[:, 1], label='$tr_{GM}$')
        axs[0, 3].plot(bin_middle_all_CLL[:, 1], hist_values_all_CLL[:, 1], label='$tr_{CLL}$')
        legend = axs[0, 3].legend()
        legend.get_frame().set_edgecolor('white')
        density_scatter(energy_MD[::4, 2], energy_MD[::4, 3], ax=axs[1, 0])
        density_scatter(energy_ML[::4, 2], energy_ML[::4, 3], ax=axs[1, 1])
        density_scatter(energy_CLL[::4, 2], energy_CLL[::4, 3], ax=axs[1, 2])
        axs[1, 3].plot(bin_middle_all_MD[:, 3], hist_values_all_MD[:, 3], label='$rot_{MD}$')
        axs[1, 3].plot(bin_middle_all_ML[:, 3], hist_values_all_ML[:, 3], label='$rot_{GM}$')
        axs[1, 3].plot(bin_middle_all_CLL[:, 3], hist_values_all_CLL[:, 3], label='$rot_{CLL}$')
        legend = axs[1, 3].legend()
        legend.get_frame().set_edgecolor('white')
        density_scatter(energy_MD[::4, 4], energy_MD[::4, 5], ax=axs[2, 0])
        density_scatter(energy_ML[::4, 4], energy_ML[::4,5], ax=axs[2, 1])
        density_scatter(energy_CLL[::4, 4], energy_CLL[::4, 5], ax=axs[2, 2])
        axs[2, 3].plot(bin_middle_all_MD[:, 5], hist_values_all_MD[:, 5], label='$tot_{MD}$')
        axs[2, 3].plot(bin_middle_all_ML[:, 5], hist_values_all_ML[:, 5], label='$tot_{GM}$')
        axs[2, 3].plot(bin_middle_all_CLL[:, 5], hist_values_all_CLL[:, 5], label='$tot_{CLL}$')
        legend = axs[2, 3].legend()
        legend.get_frame().set_edgecolor('white')

        fig.suptitle('Comparison between MD, ML and CLL results NG= ' + str(args[7]))
        # else:
        #    fig.suptitle('Comparison between MD and ML results NG= ' + str(args[5]))

        plt.savefig(args[6] + '/' + args[0] + '_Ntrain_' + str(num_train) + '_Ntest_' + str(num_test) + '_NG_' + str(
            args[7]) + '.png', dpi=300)
    ###########################################################################################################################################################
    if args[0] == 'ML-10D_rot_vel':
        av_num = 6.022e23
        conv_omega = 1.0e12  # convert [1/ps] to [1/s]
        conv_J_2_eV = 6.24e18
        if args[1] == 'H2':
            mass = 2.0158 * 0.001 / av_num
            l_b = 0.741e-10
            l_b_v = 0.741
            I = (mass / 4) * l_b ** 2
        if args[1] == 'N2':
            mass = 28.02 * 0.001 / av_num
            l_b = 1.097e-10
            l_b_v = 1.097
            I = (mass / 4) * l_b ** 2
        MD_results = np.copy(args[3])
        ML_results = np.copy(args[4])
        omega_MD = MD_results[:, 6:10] * l_b_v
        omega_pred = ML_results[:, 6:] * l_b_v

        # omega_SI_pred = args[4][:, 6:10]
        #rot_energy_in_MD = (0.5 * I * (omega_SI_MD[:, 0] ** 2 + omega_SI_MD[:, 1] ** 2)) * conv_J_2_eV
        #rot_energy_out_MD = (0.5 * I * (omega_SI_MD[:, 2] ** 2 + omega_SI_MD[:, 3] ** 2)) * conv_J_2_eV
        #rot_energy_in_pred = (0.5 * I * (omega_SI_pred[:, 0] ** 2 + omega_SI_pred[:, 1] ** 2)) * conv_J_2_eV
        #rot_energy_out_pred = (0.5 * I * (omega_SI_pred[:, 2] ** 2 + omega_SI_pred[:, 3] ** 2)) * conv_J_2_eV
        MD_results = np.concatenate(
            (MD_results[:, :6], omega_MD), axis=1)
        ML_results = np.concatenate(
            (ML_results[:, :6], omega_pred), axis=1)

        num_test = args[3].shape[0]
        num_train = args[4].shape[0]
        hist_values_all_MD = np.zeros((args[2], 10))
        bin_middle_all_MD = np.zeros((args[2], 10))
        hist_values_all_ML = np.zeros((args[2], 10))
        bin_middle_all_ML = np.zeros((args[2], 10))
        for i in range(10):
            # v_min = args[3][:, i].min()
            # v_max = args[3][:, i].max()
            hist_values, bin_edges = np.histogram(MD_results[:, i], bins=args[2], density=True)
            hist_values_ML, bin_edges_ML = np.histogram(ML_results[:, i], bins=args[2], density=True)
            hist_values_all_MD[:, i] = hist_values
            hist_values_all_ML[:, i] = hist_values_ML
            bin_middle = np.zeros((args[2], 1))
            bin_middle_ML = np.zeros((args[2], 1))
            for j in range(0, len(bin_edges) - 1):
                bin_value = (bin_edges[j] + bin_edges[j + 1]) / 2
                bin_value_ML = (bin_edges_ML[j] + bin_edges_ML[j + 1]) / 2
                bin_middle[j] = bin_value
                bin_middle_ML[j] = bin_value_ML
            bin_middle_all_MD[:, i] = bin_middle[:, 0]
            bin_middle_all_ML[:, i] = bin_middle_ML[:, 0]
        fig, axs = plt.subplots(5, 3, figsize=(9, 6), tight_layout=True)
        axs[0, 0].hist2d(MD_results[:, 0], MD_results[:, 3], bins=args[2], norm=colors.LogNorm())
        axs[0, 1].hist2d(ML_results[:, 0], ML_results[:, 3], bins=args[2], norm=colors.LogNorm())
        axs[0, 2].plot(bin_middle_all_MD[:, 3], hist_values_all_MD[:, 3], label='$Vx_{MD}$')
        axs[0, 2].plot(bin_middle_all_ML[:, 3], hist_values_all_ML[:, 3], label='$Vx_{ML}$')
        legend = axs[0, 2].legend()
        legend.get_frame().set_edgecolor('white')
        axs[1, 0].hist2d(MD_results[:, 1], MD_results[:, 4], bins=args[2], norm=colors.LogNorm())
        axs[1, 1].hist2d(ML_results[:, 1], ML_results[:, 4], bins=args[2], norm=colors.LogNorm())
        axs[1, 2].plot(bin_middle_all_MD[:, 4], hist_values_all_MD[:, 4], label='$Vy_{MD}$')
        axs[1, 2].plot(bin_middle_all_ML[:, 4], hist_values_all_ML[:, 4], label='$Vy_{ML}$')
        legend = axs[1, 2].legend()
        legend.get_frame().set_edgecolor('white')
        axs[2, 0].hist2d(MD_results[:, 2], MD_results[:, 5], bins=args[2], norm=colors.LogNorm())
        axs[2, 1].hist2d(ML_results[:, 2], ML_results[:, 5], bins=args[2], norm=colors.LogNorm())
        axs[2, 2].plot(bin_middle_all_MD[:, 5], hist_values_all_MD[:, 5], label='$Vz_{MD}$')
        axs[2, 2].plot(bin_middle_all_ML[:, 5], hist_values_all_ML[:, 5], label='$Vz_{ML}$')
        legend = axs[2, 2].legend()
        legend.get_frame().set_edgecolor('white')
        axs[3, 0].hist2d(MD_results[:, 6], MD_results[:, 8], bins=args[2], norm=colors.LogNorm())
        axs[3, 1].hist2d(ML_results[:, 6], ML_results[:, 8], bins=args[2], norm=colors.LogNorm())
        axs[3, 2].plot(bin_middle_all_MD[:, 8], hist_values_all_MD[:, 8], label='$W1_{MD}$')
        axs[3, 2].plot(bin_middle_all_ML[:, 8], hist_values_all_ML[:, 8], label='$W1_{ML}$')
        legend = axs[3, 2].legend()
        legend.get_frame().set_edgecolor('white')
        axs[4, 0].hist2d(MD_results[:, 7], MD_results[:, 9], bins=args[2], norm=colors.LogNorm())
        axs[4, 1].hist2d(ML_results[:, 7], ML_results[:, 9], bins=args[2], norm=colors.LogNorm())
        axs[4, 2].plot(bin_middle_all_MD[:, 9], hist_values_all_MD[:, 9], label='$W2_{MD}$')
        axs[4, 2].plot(bin_middle_all_ML[:, 9], hist_values_all_ML[:, 9], label='$W2_{ML}$')
        legend = axs[3, 2].legend()
        legend.get_frame().set_edgecolor('white')

        fig.suptitle('Comparison between MD and ML results NG= ' + str(args[6]))
        # else:
        #    fig.suptitle('Comparison between MD and ML results NG= ' + str(args[5]))

        plt.savefig(args[5] + '/' + args[0] + '_Ntrain_' + str(num_train) + '_Ntest_' + str(num_test) + '_NG_' + str(
            args[6]) + '.png')
    #####################################################################################################################################
    if args[0] == 'ML-10D_CLL_rot_vel':
        av_num = 6.022e23
        conv_omega = 1.0e12  # convert [1/ps] to [1/s]
        conv_J_2_eV = 6.24e18
        if args[1] == 'H2':
            mass = 2.0158 * 0.001 / av_num
            l_b = 0.741e-10
            l_b_v = 0.741
            I = (mass / 4) * l_b ** 2
        if args[1] == 'N2':
            mass = 28.02 * 0.001 / av_num
            l_b = 1.097e-10
            l_b_v = 1.097
            I = (mass / 4) * l_b ** 2
        MD_results = np.copy(args[3])
        ML_results = np.copy(args[4])
        CLL_results = np.copy(args[5])
        omega_MD = MD_results[:, 6:10] * l_b_v
        omega_pred = ML_results[:, 6:] * l_b_v
        omega_CLL = CLL_results[:, 6:] * l_b_v
        MD_results = np.concatenate(
            (MD_results[:, :6], omega_MD), axis=1)
        ML_results = np.concatenate(
            (ML_results[:, :6], omega_pred), axis=1)
        CLL_results = np.concatenate(
            (CLL_results[:, :6], omega_CLL), axis=1)
        MD_results[:, 1], MD_results[:, 4] = np.abs(MD_results[:, 1]), np.abs(MD_results[:, 4])  # in order to have same sign for scatter plot
        ML_results[:, 1], ML_results[:, 4] = np.abs(ML_results[:, 1]), np.abs(ML_results[:, 4])
        CLL_results[:, 1], CLL_results[:, 4] = np.abs(CLL_results[:, 1]), np.abs(CLL_results[:, 4])
        hist_values_all_MD,bin_middle_all_MD = conv_hist_to_bin(MD_results)
        hist_values_all_CLL, bin_middle_all_CLL = conv_hist_to_bin(CLL_results)
        hist_values_all_ML, bin_middle_all_ML = conv_hist_to_bin(ML_results)

        num_test = args[3].shape[0]
        num_train = args[4].shape[0]

        fig, axs = plt.subplots(5, 4, figsize=(12, 10), tight_layout=True)
        density_scatter(MD_results[::4, 0], MD_results[::4, 3], ax=axs[0, 0])
        density_scatter(ML_results[::4, 0], ML_results[::4, 3], ax=axs[0, 1])
        density_scatter(CLL_results[::4, 0], CLL_results[::4, 3], ax=axs[0, 2])
        axs[0, 3].plot(bin_middle_all_MD[:, 3], hist_values_all_MD[:, 3], label='$Vx_{MD}$')
        axs[0, 3].plot(bin_middle_all_ML[:, 3], hist_values_all_ML[:, 3], label='$Vx_{GM}$')
        axs[0, 3].plot(bin_middle_all_CLL[:, 3], hist_values_all_CLL[:, 3], label='$Vx_{CLL}$')
        legend = axs[0, 3].legend()
        legend.get_frame().set_edgecolor('white')
        density_scatter(MD_results[::4, 1], MD_results[::4, 4], ax=axs[1, 0])
        density_scatter(ML_results[::4, 1], ML_results[::4, 4], ax=axs[1, 1])
        density_scatter(CLL_results[::4, 1], CLL_results[::4, 4], ax=axs[1, 2])
        axs[1, 3].plot(bin_middle_all_MD[:, 4], hist_values_all_MD[:, 4], label='$Vy_{MD}$')
        axs[1, 3].plot(bin_middle_all_ML[:, 4], hist_values_all_ML[:, 4], label='$Vy_{GM}$')
        axs[1, 3].plot(bin_middle_all_CLL[:, 4], hist_values_all_CLL[:, 4], label='$Vy_{CLL}$')
        legend = axs[1, 3].legend()
        legend.get_frame().set_edgecolor('white')
        density_scatter(MD_results[::4, 2], MD_results[::4, 5], ax=axs[2, 0])
        density_scatter(ML_results[::4, 2], ML_results[::4, 5], ax=axs[2, 1])
        density_scatter(CLL_results[::4, 2], CLL_results[::4, 5], ax=axs[2, 2])
        axs[2, 3].plot(bin_middle_all_MD[:, 5], hist_values_all_MD[:, 5], label='$Vz_{MD}$')
        axs[2, 3].plot(bin_middle_all_ML[:, 5], hist_values_all_ML[:, 5], label='$Vz_{GM}$')
        axs[2, 3].plot(bin_middle_all_CLL[:, 5], hist_values_all_CLL[:, 5], label='$Vz_{CLL}$')
        legend = axs[2, 3].legend()
        legend.get_frame().set_edgecolor('white')
        density_scatter(MD_results[::4, 6], MD_results[::4, 8], ax=axs[3, 0])
        density_scatter(ML_results[::4, 6], ML_results[::4, 8], ax=axs[3, 1])
        density_scatter(CLL_results[::4, 6], CLL_results[::4, 8], ax=axs[3, 2])
        axs[3, 3].plot(bin_middle_all_MD[:, 8], hist_values_all_MD[:, 8], label='$W1_{MD}$')
        axs[3, 3].plot(bin_middle_all_ML[:, 8], hist_values_all_ML[:, 8], label='$W1_{GM}$')
        axs[3, 3].plot(bin_middle_all_CLL[:, 8], hist_values_all_CLL[:, 8], label='$W1_{CLL}$')
        legend = axs[3, 3].legend()
        legend.get_frame().set_edgecolor('white')
        density_scatter(MD_results[::4, 7], MD_results[::4, 9], ax=axs[4, 0])
        density_scatter(ML_results[::4, 7], ML_results[::4, 9], ax=axs[4, 1])
        density_scatter(CLL_results[::4, 7], CLL_results[::4, 9], ax=axs[4, 2])
        axs[4, 3].plot(bin_middle_all_MD[:, 9], hist_values_all_MD[:, 9], label='$W2_{MD}$')
        axs[4, 3].plot(bin_middle_all_ML[:, 9], hist_values_all_ML[:, 9], label='$W2_{GM}$')
        axs[4, 3].plot(bin_middle_all_ML[:, 9], hist_values_all_ML[:, 9], label='$W2_{CLL}$')
        legend = axs[4, 3].legend()
        legend.get_frame().set_edgecolor('white')

        fig.suptitle('Comparison between MD and ML results NG= ' + str(args[7]))
        # else:
        #    fig.suptitle('Comparison between MD and ML results NG= ' + str(args[5]))

        plt.savefig(args[6] + '/' + args[0] + '_Ntrain_' + str(num_train) + '_Ntest_' + str(num_test) + '_NG_' + str(
            args[7]) + '.png',dpi=300)
    #################################################################################################################################
    if args[0] == 'ML-multi':

        num_test = args[2].shape[0]
        num_train = args[3].shape[0]
        hist_values_all_MD = np.zeros((args[1], 6))
        bin_middle_all_MD = np.zeros((args[1], 6))
        hist_values_all_ML = np.zeros((args[1], 6))
        bin_middle_all_ML = np.zeros((args[1], 6))
        for i in range(6):
            v_min = args[2][:, i].min()
            v_max = args[2][:, i].max()
            hist_values, bin_edges = np.histogram(args[2][:, i], bins=args[1], range=(v_min, v_max), density=True)
            hist_values_ML, bin_edges_ML = np.histogram(args[3][:, i], bins=args[1], range=(v_min, v_max), density=True)
            hist_values_all_MD[:, i] = hist_values
            hist_values_all_ML[:, i] = hist_values_ML
            bin_middle = np.zeros((args[1], 1))
            bin_middle_ML = np.zeros((args[1], 1))
            for j in range(0, len(bin_edges) - 1):
                bin_value = (bin_edges[j] + bin_edges[j + 1]) / 2
                bin_value_ML = (bin_edges_ML[j] + bin_edges_ML[j + 1]) / 2
                bin_middle[j] = bin_value
                bin_middle_ML[j] = bin_value_ML
            bin_middle_all_MD[:, i] = bin_middle[:, 0]
            bin_middle_all_ML[:, i] = bin_middle_ML[:, 0]
        fig, axs = plt.subplots(3, 3, figsize=(9, 6), tight_layout=True)
        axs[0, 0].hist2d(args[2][:, 0], args[2][:, 3], bins=40, norm=colors.LogNorm())
        axs[0, 1].hist2d(args[3][:, 0], args[3][:, 3], bins=40, norm=colors.LogNorm())
        axs[0, 2].plot(bin_middle_all_MD[:, 3], hist_values_all_MD[:, 3], label='$Vx_{MD}$')
        axs[0, 2].plot(bin_middle_all_ML[:, 3], hist_values_all_ML[:, 3], label='$Vx_{ML}$')
        legend = axs[0, 2].legend()
        legend.get_frame().set_edgecolor('white')
        axs[1, 0].hist2d(args[2][:, 1], args[2][:, 4], bins=40, norm=colors.LogNorm())
        axs[1, 1].hist2d(args[3][:, 1], args[3][:, 4], bins=40, norm=colors.LogNorm())
        axs[1, 2].plot(bin_middle_all_MD[:, 4], hist_values_all_MD[:, 4], label='$Vy_{MD}$')
        axs[1, 2].plot(bin_middle_all_ML[:, 4], hist_values_all_ML[:, 4], label='$Vy_{ML}$')
        legend = axs[1, 2].legend()
        legend.get_frame().set_edgecolor('white')
        axs[2, 0].hist2d(args[2][:, 2], args[2][:, 5], bins=40, norm=colors.LogNorm())
        axs[2, 1].hist2d(args[3][:, 2], args[3][:, 5], bins=40, norm=colors.LogNorm())
        axs[2, 2].plot(bin_middle_all_MD[:, 5], hist_values_all_MD[:, 5], label='$Vz_{MD}$')
        axs[2, 2].plot(bin_middle_all_ML[:, 5], hist_values_all_ML[:, 5], label='$Vz_{ML}$')
        legend = axs[2, 2].legend()
        legend.get_frame().set_edgecolor('white')
        # if args[6]:
        #    fig.suptitle('Comparison between MD and ML results for' + str(args[6]) +'NG= '+str(args[5]))
        # else:
        #    fig.suptitle('Comparison between MD and ML results NG= ' + str(args[5]))
        fig.suptitle('Comparison between MD and ML results for' + str(args[6]) + 'NG= ' + str(args[5]))
        plt.savefig(args[4] + '/' + args[0] + '_Ntrain_' + str(num_train) + '_Ntest_' + str(num_test) + '_NG_' + str(
            args[5]) + '_' + str(args(6)) + '.png')
    ####################################################################################
    if args[0] == 'ML-CLL':
        num_test = args[2].shape[0]
        num_train = args[3].shape[0]
        hist_values_all_MD = np.zeros((args[1], 6))
        bin_middle_all_MD = np.zeros((args[1], 6))
        hist_values_all_ML = np.zeros((args[1], 6))
        bin_middle_all_ML = np.zeros((args[1], 6))
        hist_values_all_CLL = np.zeros((args[1], 6))
        bin_middle_all_CLL = np.zeros((args[1], 6))

        for i in range(6):
            v_min = args[2][:, i].min()
            v_max = args[2][:, i].max()
            hist_values, bin_edges = np.histogram(args[2][:, i], bins=args[1],range=(v_min, v_max), density=True)
            hist_values_ML, bin_edges_ML = np.histogram(args[3][:, i], bins=args[1],range=(v_min, v_max), density=True)
            #hist_values_CLL, bin_edges_CLL = np.histogram(args[4][:, i], bins=args[1], range=(v_min, v_max), density=True)
            #hist_values, bin_edges = np.histogram(args[2][:, i], bins=args[1], density=True)
            #hist_values_ML, bin_edges_ML = np.histogram(args[3][:, i], bins=args[1], density=True)
            hist_values_CLL, bin_edges_CLL = np.histogram(args[4][:, i], bins=args[1], density=True)
            hist_values_all_MD[:, i] = hist_values
            hist_values_all_ML[:, i] = hist_values_ML
            hist_values_all_CLL[:, i] = hist_values_CLL
            bin_middle = np.zeros((args[1], 1))
            bin_middle_ML = np.zeros((args[1], 1))
            bin_middle_CLL = np.zeros((args[1], 1))
            for j in range(0, len(bin_edges) - 1):
                bin_value = (bin_edges[j] + bin_edges[j + 1]) / 2
                bin_value_ML = (bin_edges_ML[j] + bin_edges_ML[j + 1]) / 2
                bin_value_CLL = (bin_edges_CLL[j] + bin_edges_CLL[j + 1]) / 2
                bin_middle[j] = bin_value
                bin_middle_ML[j] = bin_value_ML
                bin_middle_CLL[j] = bin_value_CLL
            bin_middle_all_MD[:, i] = bin_middle[:, 0]
            bin_middle_all_ML[:, i] = bin_middle_ML[:, 0]
            bin_middle_all_CLL[:, i] = bin_middle_CLL[:, 0]
        fig, axs = plt.subplots(3, 4, figsize=(18, 15),tight_layout=True)
        axs[0,0].hist2d(args[2][:, 0], args[2][:, 3], bins=40, norm=colors.LogNorm())
        axs[0,1].hist2d(args[3][:, 0], args[3][:, 3], bins=40, norm=colors.LogNorm())
        axs[0, 2].hist2d(args[4][:, 0], args[4][:, 3], bins=40, norm=colors.LogNorm())
        axs[0,3].plot(bin_middle_all_MD[:, 3], hist_values_all_MD[:, 3], label='$Vx_{MD}$')
        axs[0,3].plot(bin_middle_all_ML[:, 3], hist_values_all_ML[:, 3], label='$Vx_{ML}$')
        axs[0, 3].plot(bin_middle_all_CLL[:, 3], hist_values_all_CLL[:, 3], label='$Vx_{CLL}$')
        legend = axs[0,3].legend()
        legend.get_frame().set_edgecolor('white')
        axs[1, 0].hist2d(args[2][:, 1], args[2][:, 4], bins=40, norm=colors.LogNorm())
        axs[1, 1].hist2d(args[3][:, 1], args[3][:, 4], bins=40, norm=colors.LogNorm())
        axs[1, 2].hist2d(args[4][:, 1], args[4][:, 4], bins=40, norm=colors.LogNorm())
        axs[1,3].plot(bin_middle_all_MD[:, 4], hist_values_all_MD[:, 4], label='$Vy_{MD}$')
        axs[1,3].plot(bin_middle_all_ML[:, 4], hist_values_all_ML[:, 4], label='$Vy_{ML}$')
        axs[1, 3].plot(bin_middle_all_CLL[:, 4], hist_values_all_CLL[:, 4], label='$Vy_{CLL}$')
        legend = axs[1,3].legend()
        legend.get_frame().set_edgecolor('white')
        axs[2, 0].hist2d(args[2][:, 2], args[2][:, 5], bins=40, norm=colors.LogNorm())
        axs[2, 1].hist2d(args[3][:, 2], args[3][:, 5], bins=40, norm=colors.LogNorm())
        axs[2, 2].hist2d(args[4][:, 2], args[4][:, 5], bins=40, norm=colors.LogNorm())
        axs[2,3].plot(bin_middle_all_MD[:, 5], hist_values_all_MD[:, 5], label='$Vz_{MD}$')
        axs[2,3].plot(bin_middle_all_ML[:, 5], hist_values_all_ML[:, 5], label='$Vz_{ML}$')
        axs[2, 3].plot(bin_middle_all_CLL[:, 5], hist_values_all_CLL[:, 5], label='$Vz_{CLL}$')
        legend = axs[2,3].legend()
        legend.get_frame().set_edgecolor('white')
        fig.suptitle('Comparison between MD, CLL and ML results NG= '+str(args[6]))
        plt.savefig(args[5] + '/' + args[0] +'_Ntrain_'+str(num_train)+'_Ntest_'+str(num_test)+'_NG_'+str(args[6])+'.png',dpi=300)

##############################################################################################################################
def kl_divergence(p, q):
    """Function to compute the KL divergence between two distributions p and q"""
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
#####################################################################################################
def plot_KL_divergence(plot_type,vel_MD,vel_ML,num_bins,path_save_fig,N_GM):
    """Function to plot KL-divergence"""
    energy_MD = np.sum(vel_MD[:, 3:] ** 2, axis=1)
    energy_ML = np.sum(vel_ML[:, 3:] ** 2, axis=1)
    vel_MD = np.append(vel_MD,energy_MD.reshape(-1,1),axis=1)
    vel_ML = np.append(vel_ML, energy_ML.reshape(-1,1), axis=1)
    hist_values_MD = np.zeros((num_bins, 4))
    hist_values_ML = np.zeros((num_bins, 4))
    kl_diver_mine = []
    kl_diver_entropy = []
    for i in range(3):
        #v_min = np.abs(vel_MD[:, i+3]).min()
        v_min = vel_MD[:, i + 3].min()
        #v_max = np.abs(vel_MD[:, i+3]).max()
        v_max = vel_MD[:, i + 3].max()
        hist_values, bin_edges = np.histogram(vel_MD[:, i+3], bins=num_bins,range=(v_min, v_max),weights=np.ones(vel_MD.shape[0]), density=True)
        hist_values_ML1, bin_edges_ML = np.histogram(vel_ML[:, i+3], bins=num_bins,range=(v_min, v_max),weights=np.ones(vel_ML.shape[0]), density=True)
        hist_values_MD[:, i] = hist_values
        hist_values_ML[:, i] = hist_values_ML1
    energy_min = vel_MD[:, 6].min()
    energy_max = vel_MD[:, 6].max()
    hist_values_en, bin_edges_en = np.histogram(vel_MD[:, 6], bins=num_bins, range=(energy_min, energy_max),weights=np.ones(vel_MD.shape[0]), density=True)
    hist_values_ML1_en, bin_edges_ML_en = np.histogram(vel_ML[:, 6], bins=num_bins, range=(energy_min, energy_max),weights=np.ones(vel_ML.shape[0]), density=True)
    hist_values_MD[:, 3] = hist_values_en
    hist_values_ML[:, 3] = hist_values_ML1_en
    for i in range(4):
        aa = np.where(hist_values_ML[:,i] == 0)[0]
        if len(aa):
            print('There are {} zero elements in ML results in position [{}] using {} GM functions'.format(len(aa),i,N_GM))
        hist_values_ML_f=np.delete(hist_values_ML[:,i],aa)
        hist_values_MD_f=np.delete(hist_values_MD[:,i],aa)
        kl_diver_mine.append(kl_divergence(hist_values_MD_f, hist_values_ML_f))
        kl_diver_entropy.append(entropy(hist_values_MD_f,qk= hist_values_ML_f ))
    if plot_type == 'bar_plot':
        vel_types=['Vx','Vy','Vz','$V^2$']
        color_iter = itertools.cycle(['navy','red','green','blue'])
        bars = []
        lb_loc = []
        n_components_range = range(1, 2)
        labels = ['', 'GM', '','']
        fig, ax = plt.subplots()
        for i, (vel_type,color_type) in enumerate(zip(vel_types, color_iter)):
            xpos = np.array(n_components_range) + .2 * (i - 2)
            lb_loc.append(float(xpos))
            bars.append(ax.bar(xpos,kl_diver[i],width=.2, color=color_type))
        axes_lab=np.array(lb_loc)
        ax.set_xticks(axes_lab)
        ax.set_xticklabels(labels)
        ax.set_title('KL-divergence plot with'+'$N_G=$'+str(N_GM))
        ax.legend([b[0] for b in bars], vel_types)
        plt.savefig(path_save_fig+'/Bar_plot'+str(N_GM)+'.png',dpi=300)
    else:
        return kl_diver_mine,kl_diver_entropy
####################################################################################################3
def plot_KL_divergence_tot(vel_MD,vel_ML,vel_CLL,num_bins,path_save_fig,N_GM):
    """Function to plot KL-divergence"""
    energy_MD = np.sum(vel_MD[:, 3:] ** 2, axis=1)
    energy_ML = np.sum(vel_ML[:, 3:] ** 2, axis=1)
    energy_CLL = np.sum(vel_CLL[:, 3:] ** 2, axis=1)
    vel_MD = np.append(vel_MD,energy_MD.reshape(-1,1),axis=1)
    vel_ML = np.append(vel_ML, energy_ML.reshape(-1,1), axis=1)
    vel_CLL = np.append(vel_CLL, energy_CLL.reshape(-1, 1), axis=1)
    hist_values_MD = np.zeros((num_bins, 4))
    hist_values_ML = np.zeros((num_bins, 4))
    hist_values_CLL = np.zeros((num_bins, 4))
    kl_diver_mine_ML = []
    kl_diver_entropy_ML = []
    kl_diver_mine_CLL = []
    kl_diver_entropy_CLL = []
    for i in range(3):
        #v_min = np.abs(vel_MD[:, i+3]).min()
        v_min = vel_MD[:, i + 3].min()
        #v_max = np.abs(vel_MD[:, i+3]).max()
        v_max = vel_MD[:, i + 3].max()
        hist_values, bin_edges = np.histogram(vel_MD[:, i+3], bins=num_bins,range=(v_min, v_max),weights=np.ones(vel_MD.shape[0]), density=True)
        hist_values_ML1, bin_edges_ML = np.histogram(vel_ML[:, i+3], bins=num_bins,range=(v_min, v_max),weights=np.ones(vel_ML.shape[0]), density=True)
        hist_values_CLL1, bin_edges_CLL = np.histogram(vel_CLL[:, i + 3], bins=num_bins, range=(v_min, v_max),weights=np.ones(vel_CLL.shape[0]), density=True)
        hist_values_MD[:, i] = hist_values
        hist_values_ML[:, i] = hist_values_ML1
        hist_values_CLL[:, i] = hist_values_CLL1
    energy_min = vel_MD[:, 6].min()
    energy_max = vel_MD[:, 6].max()
    hist_values_en, bin_edges_en = np.histogram(vel_MD[:, 6], bins=num_bins, range=(energy_min, energy_max),weights=np.ones(vel_MD.shape[0]), density=True)
    hist_values_ML1_en, bin_edges_ML_en = np.histogram(vel_ML[:, 6], bins=num_bins, range=(energy_min, energy_max),weights=np.ones(vel_ML.shape[0]), density=True)
    hist_values_CLL1_en, bin_edges_CLL_en = np.histogram(vel_CLL[:, 6], bins=num_bins, range=(energy_min, energy_max),weights=np.ones(vel_CLL.shape[0]), density=True)
    hist_values_MD[:, 3] = hist_values_en
    hist_values_ML[:, 3] = hist_values_ML1_en
    hist_values_CLL[:, 3] = hist_values_CLL1_en
    for i in range(4):
        aa = np.where(hist_values_ML[:,i] == 0)[0]
        if len(aa):
            print('There are {} zero elements in ML results in position [{}] using {} GM functions'.format(len(aa),i,N_GM))
        hist_values_ML_f=np.delete(hist_values_ML[:,i],aa)
        hist_values_MD_f=np.delete(hist_values_MD[:,i],aa)
        hist_values_CLL_f = np.delete(hist_values_CLL[:, i], aa)
        kl_diver_mine_ML.append(kl_divergence(hist_values_MD_f, hist_values_ML_f))
        kl_diver_entropy_ML.append(entropy(hist_values_MD_f,qk= hist_values_ML_f ))
        kl_diver_mine_CLL.append(kl_divergence(hist_values_MD_f, hist_values_CLL_f))
        kl_diver_entropy_CLL.append(entropy(hist_values_MD_f, qk=hist_values_CLL_f))
    if plot_type == 'bar_plot':
        vel_types=['Vx','Vy','Vz','$V^2$']
        color_iter = itertools.cycle(['navy','red','green','blue'])
        bars = []
        lb_loc = []
        n_components_range = range(1, 2)
        labels = ['', 'GM', '','','','CLL','','']
        fig, ax = plt.subplots()
        for i, (vel_type,color_type) in enumerate(zip(vel_types, color_iter)):
            xpos = np.array(n_components_range) + .2 * (i - 2)
            lb_loc.append(float(xpos))
            bars.append(ax.bar(xpos,kl_diver_mine_ML[i],width=.2, color=color_type))
        axes_lab=np.array(lb_loc)
        ax.set_xticks(axes_lab)
        ax.set_xticklabels(labels)
        ax.set_title('KL-divergence plot with'+'$N_G=$'+str(N_GM))
        ax.legend([b[0] for b in bars], vel_types)
        plt.savefig(path_save_fig+'/Bar_plot'+str(N_GM)+'.png',dpi=300)
    else:
        return kl_diver_mine,kl_diver_entropy





    #plt.figure()




