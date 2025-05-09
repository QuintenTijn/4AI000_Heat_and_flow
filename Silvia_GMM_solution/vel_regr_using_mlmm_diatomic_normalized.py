import os
import numpy as np
from scipy.special import erf, erfinv
#from math import erf
from numpy.linalg import norm
from mlmm import VelocityData, plot_learn_curve, n_fold_cv, fileinfo,VelocityDataOmegaData, OmegaData
#from sklearn.kernel_ridge import KernelRidge
#from sklearn.gaussian_process import GaussianProcessRegressor as GPR
#from sklearn.gaussian_process.kernels import RBF,ConstantKernel as C
#from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.mixture import GaussianMixture as GM
#from sklearn.mixture import BayesianGaussianMixture as BGM
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from cycler import cycler
import pyprind
from datetime import date
today = date.today().strftime("%b-%d-%Y")
import periodictable
from scipy import constants as cs
import sys
from mlmm import func_postprocess
import time
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
#import seaborn as sns
import pandas as pd
import itertools


er_type = 'MAE'
#
#-- For training based on translational velocities system_6D =True
#---- For training based on translational+ rotational velocities system_10D =True
system_6D = False
system_10D = True
system_omega = True
#-----Wall temperature is needed to compute velocities for CLL kernel
wall_temp = 300
Liao_Transfer_Function = True

kB,conv_v,conv_omega,av_num = 1.38064852e-23,1.0e2,1.0e12,6.022e23


GM_nofgaussians=10
reg_list = [GM(n_components=GM_nofgaussians,tol=0.001)]

x_data = []
y_data = []

x_omega = []
y_omega = []

x_MD = []
y_MD = []

path_to_data = r'C:\Users\20180384\OneDrive - TU Eindhoven\Desktop\PhD\Assignment_ML\New\Assginment_ML\Assginment_ML\Solutions\H2'
#path_save_data='./'+path_to_data
#path_to_data = 'Shahin/test_data/'



for file in os.listdir(path_to_data):

    if file.endswith(".txt") and "MD" in file:
        x_MD.append(os.path.join(path_to_data,file))
        y_MD.append('')

    if file.endswith(".txt") and "omega" in file:
        x_omega.append(os.path.join(path_to_data,file))
        y_omega.append('')
print("Data files found:",x_MD)

######################################################################################
#------------------------------- Getting translational velocity data
####--------------------------------------------------------------------------------------------------
for x_data_file,y_data_file in zip(x_MD,y_MD):
    
    #conf = VelocityData(x_data_file, frames=frames)#[0,30000])
    conf = VelocityData(x_data_file, frames=None)#[0,30000])
    conf.getRep(rep='vxvyvz',nuc=None) #rep options: vxvyvz, vel2norm, vx2,vy2,vz2,vx,vy,vz
    file_name=x_data_file.replace('.','/')
    file_name=file_name.split('/')
    set_name = file_name[1]
    X = conf.X
    y = conf.y
    if "He" in path_to_data:
        mass = getattr(periodictable,'He').mass
        gas_name = 'He'
    elif "Ar" in path_to_data:
        mass = getattr(periodictable,'Ar').mass
        gas_name = 'Ar'
    elif "H2" in path_to_data:
        mass =2* getattr(periodictable,'H').mass
        gas_name = 'H2'
        l_b=0.741e-10
        mass_kg = mass * 0.001 / av_num
        I=(mass_kg/4)*l_b**2
    elif "N2" in path_to_data:
        mass =2* getattr(periodictable,'N').mass
        gas_name = 'N2'
        l_b=1.097e-10
        mass_kg = mass * 0.001 / av_num
        I=(mass_kg/4)*l_b**2
    else:
        print ("Unable to identify impinging atom type...")
        ele = str(input("Enter the impinging atom symbol : "))
        mass = getattr(periodictable,ele).mass
        v_mp = np.sqrt(2 * kB * wall_temp / mass_kg)
        omega_mp = np.sqrt(2 * kB * wall_temp / I)
    print ('Atomic mass for {} molecule is {:1.4f} \n'.format(gas_name,mass))
    n_MD = X.shape[0]
    print('Number of MD tr velocity data points: {} \n'.format(n_MD))
    v_TF,T_in,T_out,theta_in,theta_out=func_postprocess.liao_transform(X,mass,'y')
    v_RTF=func_postprocess.liao_R_transform(v_TF,theta_in,theta_out,'y')
    MAC_x,MAC_y,MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr=func_postprocess.compute_AC_correlation_method(X,'Ar','y') #
    AC_MD_tr = [MAC_x,MAC_y,MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr]
    print('For '+gas_name+' the computed ACs from translational MD are:\n')
    print('MAC_x={:1.4f} MAC_y={:1.4f} MAC_z={:1.4f} EAC_x={:1.4f} EAC_y={:1.4f} EAC_z={:1.4f}' \
          ' EAC_tot={:1.4f}\n'.format(MAC_x,MAC_y, 
          MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr)) # flot number formatting value:width.percision
    MAC_x_RTF,MAC_y_RTF,MAC_z_RTF,EAC_x_RTF,EAC_y_RTF,EAC_z_RTF,\
    EAC_tr_RTF=func_postprocess.compute_AC_correlation_method(v_RTF,'Ar','y')
    print('For '+gas_name+' the computed ACs from translational MD after R-transfer function are:\n')
    print('MAC_x={:1.4f} MAC_y={:1.4f} MAC_z={:1.4f} EAC_x={:1.4f} EAC_y={:1.4f} EAC_z={:1.4f}' \
          ' EAC_tot={:1.4f}\n'.format(MAC_x_RTF,MAC_y_RTF, 
                     MAC_z_RTF,EAC_x_RTF,EAC_y_RTF,EAC_z_RTF,EAC_tr_RTF)) 

######################################################################################
#------------------------------- Getting angular velocity data
####--------------------------------------------------------------------------------------------------
if system_omega:
    for x_data_file2,y_data_file2 in zip(x_omega,y_omega):
        #conf2 = OmegaData(x_data_file2, frames=frames)#[0,30000])
        conf2 = OmegaData(x_data_file2, frames=None)#[0,30000])
        conf2.getRep(rep='omega1omega2',nuc=None) #rep options: vxvyvz, vel2norm, vx2,vy2,vz2,vx,vy,vz
        omega_file_name = x_data_file2.replace('.','/')
        omega_file_name = omega_file_name.split('/')
        omega_name = omega_file_name[1]
        print(omega_name)
        X2 = conf2.X2
        y2 = conf2.y2
        print('Number of MD rot velocity data points: {} \n'.format(X2.shape[0]))
        X2_TF=np.copy(X2)
        Y2_TF=np.copy(y2)
        omega_TF=np.vstack((X2_TF,-X2_TF))
        y2_TF=np.hstack((Y2_TF,-Y2_TF))
        print('Mean omega_in_1: {:1.4f}'.format(np.mean(X2[:,1])))
        print('Mean omega_in_1: {:1.4f}'.format(np.mean(omega_TF[:,1])))
        




    
    


######################################################################################
#------------------------------- Implementing GM model on 6D data
####--------------------------------------------------------------------------------------------------
   
if system_6D:
    vel_cll = func_postprocess.velocity_CLL(X, wall_temp, gas_name)
    MAC_x_CLL,MAC_y_CLL,MAC_z_CLL,EAC_x_CLL,EAC_y_CLL,EAC_z_CLL,\
             EAC_tr_CLL=func_postprocess.compute_AC_correlation_method(vel_cll,'Ar','y')
    AC_CLL = [MAC_x_CLL,MAC_y_CLL,MAC_z_CLL,EAC_x_CLL,EAC_y_CLL,EAC_z_CLL,\
             EAC_tr_CLL]
    AC = ['MAC_x','MAC_y','MAC_z','EAC_x','EAC_y','EAC_z','EAC_tr']
    AC_diff_CLL=func_postprocess.ac_diff(AC_MD_tr,AC_CLL)
    ###########################################################################
    if Liao_Transfer_Function:
        data_train = np.copy(v_TF[:n_MD,:]) # To avoid mean shift in the case of Couette flow
        print('The total number of training points is: {} \n'.format(data_train.shape[0]))
    else:
        data_train = np.copy(X)
    print('The total number of training points is: {} \n'.format(data_train.shape[0]))

    GM_nofgaussians = range(10,20, 50) # We can define a loop for training based on different number of Guassians
    # In this specific example NG =10
    for NG in GM_nofgaussians:
        reg_list = [GM(n_components=NG, tol=0.001)]
        for reg in reg_list:
             start_time = time.time()
             info = fileinfo(reg=reg,xdata=x_data_file, ydata=y_data_file, conf=conf)
             if info.reg_name in ("GaussianMixture","BayesianGaussianMixture"):
                 flog = open (info.data_fname,"a")
                 flog.write("%s"%(today))
                 reg.fit(data_train)
                 pred = reg.sample(n_samples=len(data_train))
                 print(data_train.shape,pred[0].shape)
                 v_pred=np.copy(pred[0])
                 end_time = time.time()
                 sim_time = (end_time - start_time) / 60
                 if Liao_Transfer_Function:
                    v_RTF_pred=func_postprocess.liao_R_transform(v_pred,theta_in,theta_out,'y') # using RTF on predicted results
                    MAC_x_ML,MAC_y_ML,MAC_z_ML,EAC_x_ML,EAC_y_ML,EAC_z_ML,\
                                 EAC_tr_ML=func_postprocess.compute_AC_correlation_method(v_RTF_pred,'Ar','y')
                    AC_ML = [MAC_x_ML,MAC_y_ML,MAC_z_ML,EAC_x_ML,EAC_y_ML,EAC_z_ML,\
                                 EAC_tr_ML]
       ########################################################################################             


                #####################################################################################            
                 with open (path_to_data+"\pred_velocities_RTF_6D_NG_"+str(NG)+".txt","w") as fp:
                    # Predicted velocities from GM after RTF
                    fp.write("Vx_in, Vy_in, Vz_in, Vx_out, Vy_out, Vz_out\n")
                    for elem in v_RTF_pred:
                               fp.write("%f %f %f %f %f %f\n"%(elem[0],elem[1],elem[2],\
                                                               elem[3],elem[4],elem[5]))
             #############################################################################################
                 with open (info.directory + '/CLL_velocities_6D.txt',"w") as fp:
                    # Predicted velocities from GM after RTF
                    fp.write("Vx_in, Vy_in, Vz_in, Vx_out, Vy_out, Vz_out\n")
                    for elem in vel_cll:
                               fp.write("%f %f %f %f %f %f\n"%(elem[0],elem[1],elem[2],\
                                                               elem[3],elem[4],elem[5]))                       
             ##########################################################################################3  

                 AC_diff_ML=func_postprocess.ac_diff(AC_MD_tr,AC_ML)                    
                 with open (info.directory + '/computed_ACs.txt', "a") as fp:
                    fp.write("%s \n"%(today))
                    fp.write('Used data set: {}  \n'.format(set_name))
                    fp.write("Number of used GM functions is: %s \n" % (GM_nofgaussians))
                    fp.write('Liao TF has been implemented: {}\n'.format(str(Liao_Transfer_Function)))
                    fp.write('6D data has been used \n')
                    fp.write('Simulation time is {:1.4f} minutes \n'.format(sim_time))
                    fp.write(100 * '*')
                    fp.write('\n')
                    fp.write('ACs obtained from MD results (without TF) for '+ gas_name+':\n')
                    fp.write('MAC_x={:1.4f} MAC_y={:1.4f} MAC_z={:1.4f} EAC_x={:1.4f} EAC_y={:1.4f} EAC_z={:1.4f}' \
                       ' EAC_tot={:1.4f}\n'.format(MAC_x,MAC_y, 
                            MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr))
                    fp.write('ACs obtained from CLL for '+ gas_name+':\n')

                    ##################################################################################
                    for (ac_label,ac_cll,ac_cll_dif) in zip (AC,AC_CLL,AC_diff_CLL):
                        fp.write('{}={:1.4f}({:1.2f}%), '.format(ac_label,ac_cll,ac_cll_dif))
                    fp.write('\n')
                    fp.write('ACs obtained from the ML model for '+ gas_name+':\n')
                    for (ac_label,ac_ml,ac_ml_dif) in zip (AC,AC_ML,AC_diff_ML):
                        fp.write('{}={:1.4f}({:1.2f}%), '.format(ac_label,ac_ml,ac_ml_dif))
                    fp.write('\n')
                    fp.write(100 * '#')
                    fp.write('\n')
    
######################################################################################
#------------------------------- Implementing GM model on 10D data
####--------------------------------------------------------------------------------------------------                   
                  
if system_10D:
    vel_tr = np.copy(X)
    vel_tr[:,1]=np.abs(vel_tr[:,1])
    vel_tr[:,4]=np.abs(vel_tr[:,4])
    data_for_AC_MD = np.concatenate((vel_tr,X2),axis=1)
    data_for_AC_MD_RTF = np.concatenate((v_RTF[:n_MD,:],X2),axis=1)
    MAC_x,MAC_y,MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr,EAC_tr_energy,\
                EAC_rot,EAC_rot_energy,\
             EAC_tot_energy=func_postprocess.compute_AC_correlation_method(data_for_AC_MD,gas_name,'y')
    MAC_x_R, MAC_y_R, MAC_z_R, EAC_x_R, EAC_y_R, EAC_z_R, EAC_tr_R, EAC_tr_energy_R, \
    EAC_rot_R, EAC_rot_energy_R, \
    EAC_tot_energy_R = func_postprocess.compute_AC_correlation_method(data_for_AC_MD_RTF, gas_name, 'y')
    AC_MD_tot = [MAC_x,MAC_y,MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr,EAC_rot,EAC_tot_energy]
    AC_MD_tot_R = [MAC_x_R, MAC_y_R, MAC_z_R, EAC_x_R, EAC_y_R, EAC_z_R, EAC_tr_R, EAC_rot_R, EAC_tot_energy_R]
    vel_cll_10D = func_postprocess.velocity_CLL(data_for_AC_MD,wall_temp,gas_name)
    MAC_x_CLL,MAC_y_CLL,MAC_z_CLL,EAC_x_CLL,EAC_y_CLL,EAC_z_CLL,EAC_tr_CLL,EAC_tr_energy_CLL,\
                EAC_rot_CLL,EAC_rot_energy_CLL,\
             EAC_tot_energy_CLL=func_postprocess.compute_AC_correlation_method(vel_cll_10D,gas_name,'y')
    AC_CLL_tot = [MAC_x_CLL,MAC_y_CLL,MAC_z_CLL,EAC_x_CLL,EAC_y_CLL,EAC_z_CLL,EAC_tr_CLL,\
                EAC_rot_CLL,EAC_tot_energy_CLL]
    AC_diff_CLL=func_postprocess.ac_diff(AC_MD_tot,AC_CLL_tot)
    AC_diff_CLL_R = func_postprocess.ac_diff(AC_MD_tot_R, AC_CLL_tot)
    AC_labeles = ['MAC_x','MAC_y','MAC_z','EAC_x','EAC_y','EAC_z','EAC_tr',\
                'EAC_rot','EAC_tot']
    
    if Liao_Transfer_Function:
        #omega_bond = np.copy(X2)
        v_tr=np.copy(v_TF[:n_MD,:])
        v_omega = np.copy(X2)
        v_mp = np.sqrt(2 * kB * wall_temp / mass_kg)
        omega_mp = np.sqrt(2 * kB * wall_temp / I)
        data_train_10D = np.concatenate((v_tr*conv_v/v_mp,X2*conv_omega/omega_mp),axis=1)
        #data_train_10D = np.concatenate((v_TF[:n_MD,:],X2),axis=1)
        print('The total number of training points is: {} \n'.format(data_train_10D.shape[0]))
        #func_postprocess.plot_pdf_tr_velocities_hist('MD-diatomic',50,data_train_10D,path_save_data)
    else:
        data_train_10D = data_for_AC_MD
        print('The total number of training points is: {} \n'.format(data_train_10D.shape[0]))
        
    
    
    GM_nofgaussians = range(10,20, 50) # We can define a loop for training based on different number of Guassians
    # In this specific example NG =10
    for NG in GM_nofgaussians:
        reg_list = [GM(n_components=NG, tol=0.001)]
        for reg in reg_list:
             start_time = time.time()
             info = fileinfo(reg=reg,xdata=x_data_file, ydata=y_data_file, conf=conf)
             if info.reg_name in ("GaussianMixture","BayesianGaussianMixture"):
                 flog = open (info.data_fname,"a")
                 flog.write("%s"%(today))
                 #alpha_vs_m(info.res_name.replace("pred","alpha_m"), X, mix=np.arange(20,250,20))
                 #flog.write(",%s"%(os.path.split(info.res_name.replace("pred","alpha_m"))[1]))
                 reg.fit(data_train_10D)
                 pred = reg.sample(n_samples=len(data_train_10D))
                 vel_pred = pred[0]
                 print(data_train_10D.shape,vel_pred.shape)
                 v_pred=np.copy(pred[0])
                 bic_value = reg.bic(data_train_10D)
                 end_time = time.time()
                 sim_time = (end_time - start_time) / 60
                 if Liao_Transfer_Function:
                     v_pred[:,:6] = (v_pred[:,:6]*v_mp)/conv_v
                     v_pred[:,6:] = (v_pred[:,6:]*omega_mp)/conv_omega
                     v_RTF_pred=func_postprocess.liao_R_transform(v_pred[:,:6],theta_in,theta_out,'y') # using RTF on predicted results
                     v_RTF_pred = np.append(v_RTF_pred,v_pred[:,6:],axis=1)
                     MAC_x_ML,MAC_y_ML,MAC_z_ML,EAC_x_ML,EAC_y_ML,EAC_z_ML,EAC_tr_ML,EAC_tr_energy_ML,\
                                EAC_rot_ML,EAC_rot_energy_ML,\
                     EAC_tot_energy_ML=func_postprocess.compute_AC_correlation_method(v_RTF_pred,gas_name,'y')
                     AC_ML = [MAC_x_ML,MAC_y_ML,MAC_z_ML,EAC_x_ML,EAC_y_ML,EAC_z_ML,EAC_tr_ML,
                                EAC_rot_ML,EAC_tot_energy_ML]
                     AC_diff_ML=func_postprocess.ac_diff(AC_MD_tot,AC_ML)
                     AC_diff_ML_RTF = func_postprocess.ac_diff(AC_MD_tot_R, AC_ML)
         #######################################################################################################                
        ########################################################################################################                               
                     
                     with open (path_to_data+"\pred_velocities_RTF_10D_NG_"+str(NG)+".txt","w") as fp:
                         fp.write("Vx_in, Vy_in, Vz_in, Vx_out, Vy_out, Vz_out, omegin1, omegin2, omegout1, omegout2\n")
                         for elem in v_RTF_pred:
                               fp.write("%f %f %f %f %f %f %f %f %f %f\n"%(elem[0],elem[1],elem[2],\
                                         elem[3],elem[4],elem[5], elem[6], elem[7], elem[8], elem[9]))
         ############################################################################################# 
                     with open (info.directory + '/CLL_velocities_10D.txt',"w") as fp:
                        # Predicted velocities from GM after RTF
                        fp.write("Vx_in, Vy_in, Vz_in, Vx_out, Vy_out, Vz_out, omegin1, omegin2, omegout1, omegout2\n")
                        for elem in vel_cll_10D:
                                   fp.write("%f %f %f %f %f %f %f %f %f %f\n"%(elem[0],elem[1],elem[2],\
                                         elem[3],elem[4],elem[5], elem[6], elem[7], elem[8], elem[9]))   
                                       
                     # The following plot commands can be excluded in order to accelerate the performance
                     func_postprocess.plot_pdf_tr_velocities_hist('ML-10D_CLL_rot_vel','H2',30, data_for_AC_MD,v_RTF_pred,\
                                                                 vel_cll_10D ,path_to_data,NG)
                     func_postprocess.plot_pdf_tr_velocities_hist('ML-10D_CLL_rot_En','H2',30, data_for_AC_MD,v_RTF_pred,\
                                                                 vel_cll_10D ,path_to_data,NG)
                     func_postprocess.plot_pdf_tr_velocities_hist('ML-10D_CLL_only_energy','H2',30, data_for_AC_MD,v_RTF_pred,\
                                                                 vel_cll_10D ,path_to_data,NG)
                         
        
                     
        #########################################################################################################                               
                     with open (info.directory + '/computed_ACs.txt', "a") as fp:
                         fp.write("%s \n"%(today))
                         fp.write('Used data set: {}  \n'.format(set_name))
                         fp.write("Number of used GM functions is: %s \n" % (NG))
                         fp.write('The value of BIC is: {:1.4f}\n'.format(bic_value))
                         fp.write('Liao TF has been implemented: {}\n'.format(str(Liao_Transfer_Function)))
                         fp.write('10D data has been used \n')
                         fp.write('Simulation time is {:1.4f} minutes \n'.format(sim_time))
                         fp.write(100 * '*')
                         fp.write('\n')
                         fp.write('ACs obtained from MD results (without TF) for '+ gas_name+':\n')
                         fp.write('MAC_x={:1.4f} MAC_y={:1.4f} MAC_z={:1.4f} EAC_x={:1.4f} EAC_y={:1.4f} EAC_z={:1.4f}' \
                            ' EAC_tr_vel={:1.4f}, EAC_tr={:1.4f},EAC_rot_vel={:1.4f},EAC_rot={:1.4f},'\
                            'EAC_tot={:1.4f}\n'.format(MAC_x,MAC_y,MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr,\
                                   EAC_tr_energy,EAC_rot,EAC_rot_energy,EAC_tot_energy))
                         fp.write('ACs obtained from MD results (after RTF) for ' + gas_name + ':\n')
                         fp.write('MAC_x={:1.4f} MAC_y={:1.4f} MAC_z={:1.4f} EAC_x={:1.4f} EAC_y={:1.4f} EAC_z={:1.4f}' \
                                  ' EAC_tr_vel={:1.4f}, EAC_tr={:1.4f},EAC_rot_vel={:1.4f},EAC_rot={:1.4f},' \
                                  'EAC_tot={:1.4f}\n'.format(MAC_x_R, MAC_y_R, MAC_z_R, EAC_x_R, EAC_y_R, EAC_z_R, EAC_tr_R, \
                                                             EAC_tr_energy_R, EAC_rot_R, EAC_rot_energy_R, EAC_tot_energy_R))
                         fp.write('ACs obtained from CLL model for '+ gas_name+':\n')
                         for ac_lab,ac,ac_diff in zip(AC_labeles,AC_CLL_tot,AC_diff_CLL):
                             fp.write('{}={:1.4f}({:1.2f}%),'.format(ac_lab,ac,ac_diff))
                         fp.write('\n')
                         ###########################################################################
                         fp.write('ACs obtained from the ML model for '+ gas_name+' MD:\n')
                         for ac_lab,ac,ac_diff in zip(AC_labeles,AC_ML,AC_diff_ML):
                             fp.write('{}={:1.4f}({:1.2f}%),'.format(ac_lab,ac,ac_diff))
                         fp.write('\n')
                         fp.write('ACs obtained from the ML model for '+ gas_name+' MD-RTF:\n')
                         for ac_lab,ac,ac_diff in zip(AC_labeles,AC_ML,AC_diff_ML_RTF):
                             fp.write('{}={:1.4f}({:1.2f}%),'.format(ac_lab,ac,ac_diff))
                         fp.write('\n')
                         fp.write(100 * '#')
                         fp.write('\n')
         ########################################---Without TF
       



            
    
                          
                




            

