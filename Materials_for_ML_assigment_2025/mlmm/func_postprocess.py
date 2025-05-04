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



