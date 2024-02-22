# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:56:47 2024

@author: sacco004
"""

import cv2
import numpy as np
import easygui as egui
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter as sg_filter
from scipy.optimize import curve_fit
import time

def file_finder(directory,requirement):
    '''Searches in a directory and its subdirectories for csv files with a specific requirement in their name'''
    data_files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(".tif") and (requirement in filename.lower()):
                data_files.append(os.path.join(dirpath, filename))
    return data_files

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def Gaussian(x,C0,R0,u,y0):
    '''Gaussian equation to fit'''
    
    return y0+C0*np.exp(-2*(x-u)**2/R0**2)

def linear(x,m,c):
    return m*x+c

folder_all= egui.diropenbox("Select Folder to Import TIFF Images of a stack")
os.chdir(f'{folder_all}')
files = file_finder(folder_all, 'frap') 

#Import images from different samples
samples = set([file.split('\\')[-1].rsplit('_',2)[0].rstrip('.tif') for file in files])
stacks = {sample:{} for sample in samples}
subtracted_stacks = {sample:{} for sample in samples}

for sample in samples:
    for file in files:
        if sample in file:
            name_image = file.split('\\')[-1].split('_')[3].rstrip('.tif')
            x_axis_length = file.split('\\')[-1].split('_')[2].rstrip('um').split('-') #micro-m
            
            image = cv2.imread(file)
            
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Convert the grayscale image to float64
            gray_image = np.asarray(gray_image, dtype=np.float64)
            
            profile = gray_image.mean(1)
            # Add the grayscale image to the stack
            stacks[sample][name_image] = profile
    
    
    x_axis_length = float(f"{x_axis_length[0]}.{x_axis_length[1]}")
    x_axis = np.linspace(0, x_axis_length, len(profile))
    
    pre_bleach_images = [[stacks[sample][x] for x in stacks[sample] if int(x.split('_')[-1]) in np.arange(1,6)],]
    pre_bleach_images = np.concatenate(pre_bleach_images,axis=1).T
    average_pre_bleach = pre_bleach_images.mean(1)
    
    keys_to_delete = [x for x in stacks[sample] if int(x.split('_')[-1]) in np.arange(1, 6)]
    
    for key in keys_to_delete:
        del stacks[sample][key]

    #Store subtracted profiles
    
    subtracted_stacks[sample] = pd.DataFrame({r"Distance, um": x_axis,'Mean_pre_bleach': average_pre_bleach})
    
    for stack_key, stack_value in stacks[sample].items():
    
        column_suffix = int(stack_key)
        subtracted_stack = pd.DataFrame({f"PB_{column_suffix}": np.subtract(sg_filter(average_pre_bleach, 5, 3),sg_filter(stack_value, 5, 3))})
        subtracted_stacks[sample] = pd.concat([subtracted_stacks[sample],subtracted_stack],axis=1)

#Calculate amplitudes of Gaussians over time
plants = list(set([plant.split('_')[1]for plant in samples]))
time_axis=list(np.arange(1,61,1)) + list(np.arange(61,360,5))
amplitudes=pd.DataFrame(time_axis,columns=['Time, s'])
radii=pd.DataFrame(time_axis,columns=['Time, s'])

iteration = 0
total = sum([len(subtracted_stacks[sample].columns[1:]) for sample in samples])
for sample in samples:
    
    a=[]
    r=[]
        
    #Determine initial center of the bleach
    p0= [1,1,3,0]
    bleach_position, pcov = curve_fit(Gaussian, subtracted_stacks[sample]['Distance, um'], subtracted_stacks[sample]['PB_6'], p0=p0,bounds=((0,0,0,-np.inf),(np.inf,np.inf,np.inf,np.inf)))
    
    #Use the bleach position to fit all the next post-bleach lines

    for stack in subtracted_stacks[sample].columns[2:]:
        
        if iteration == 0: 
            begin = time.time() 
        
        #Perform fit to Gaussian
        p0= [1,1,0] # Initial guess of C0,R0,u,y0
        popt, pcov = curve_fit(lambda x,C0,R0,y0: Gaussian(x,C0,R0,bleach_position[2],y0), subtracted_stacks[sample]['Distance, um'], subtracted_stacks[sample][stack], p0=p0,bounds=((0,0,-np.inf),(np.inf,np.inf,np.inf)))
        a.append(popt[0])
        r.append(popt[1])
    
        # Get the standard deviations of the parameters
        perr = np.sqrt(np.diag(pcov))
    
        # Construct the lower and upper bounds for each parameter
        lower_bounds = popt - 1.96 * perr  # 1.96 corresponds to a 95% confidence interval
        upper_bounds = popt + 1.96 * perr
    
        # Generate x values for the fitted curve
        x_fit = np.linspace(min(subtracted_stacks[sample]['Distance, um']), max(subtracted_stacks[sample]['Distance, um']), 1000)
        # plt.plot(subtracted_stacks[sample]['Distance, um'],Gaussian(subtracted_stacks[sample]['Distance, um'],*[*popt[:2],bleach_position[2],popt[-1]]))
        # plt.scatter(subtracted_stacks[sample]['Distance, um'],subtracted_stacks[sample][stack],label = stack)
        # plt.legend()
        # plt.show()
        
        if iteration == 0: 
            end = time.time() 
            
        remaining_time = (end-begin)*(total-iteration)
        printProgressBar(iteration, total-1, prefix = 'Calculating Gaussian fitting parameters:', suffix = f'Time remaining: {remaining_time:.1f} s', length = 50)
        iteration+=1
        
        
    amplitudes[sample]=a
    radii[sample]=r
    
for plant in plants:
    amplitudes[f'Mean_{plant}']= amplitudes[[col for col in amplitudes.columns[1:] if plant == col.split('_')[1]]].mean(axis=1)
    amplitudes[f'SEM_{plant}']= amplitudes[[col for col in amplitudes.columns[1:] if plant == col.split('_')[1]]].sem(axis=1)
      
#Plot amplitudes and radii
for sample in samples:
    
    fig,ax1 = plt.subplots(figsize=(5,3.5),layout='tight')
    fig.suptitle(sample)
    ax1.scatter(amplitudes['Time, s'],amplitudes[sample],color='b',label='C')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('C',color='b')
    ax1.tick_params(axis ='y', labelcolor = 'b') 
    
    ax2 = ax1.twinx() 
    ax2.scatter(radii['Time, s'],radii[sample],color='r',label='R')
    ax2.set_ylabel('R',color='r')
    ax2.tick_params(axis ='y', labelcolor = 'r') 
    # fig.legend(loc=(0.75,0.75),frameon=False)
    plt.savefig(f'C&R_{sample}.svg')
    plt.show()
    


#Calculate and plot diffusion coefficients. Calculated over the first minute of post-bleach

D_coeffs=pd.DataFrame(columns=plants)

for plant in plants:
    D=[]
    fig, ax = plt.subplots(figsize=(4.5,3.5),layout='tight')
    fig.suptitle(f'{plant}')
    ax.set_xlabel('Time, s')
    ax.set_ylabel('(C$_{0,0}$/C$_{0,t}$)$^2$')
    
    for sample in samples:
        if plant == sample.split('_')[1]:
            
            x=amplitudes['Time, s'][:60]
            y=list((amplitudes[sample][0]/amplitudes[sample])**2)[:60]
            ax.scatter(x,y)
            popt,pcov = curve_fit(linear, x, y)
            ax.plot(x, linear(np.array(x),*popt),color='k')
            
            '''Here I calculate the diffusion coefficient from the linear regression:
                D = (R0)^2*m/8
                radii[0] is the initial radius of the bleach
                popt[0] is m
                *1e-8 is to convert it from um2 to cm2
                .2E formats it to scientific notation
                '''
            d=radii[sample][0]**2*popt[0]*1e-8/8
            D.append(d)
    D = pd.Series(D)
    D_coeffs[plant]=D
    
    ax.text(0.1,0.8,f"{np.mean(D):.2E} cm$^2$ s$^{-1}$", transform=ax.transAxes)
        
    fig.savefig(f'Diffusion plot_{plant}.svg')
    plt.show()


#Calculate propagated errors
for plant in list(set([plant.split('_')[1]for plant in samples])):
    gradient_list=[]
    SEM_prop_list=[]
    
    for i in range(len(amplitudes[f'Mean_{plant}'])):
        gradient = (amplitudes.loc[0,f'Mean_{plant}']/amplitudes.loc[i,f'Mean_{plant}'])**2
        gradient_list.append(gradient)
        
        sepropagated = (2*amplitudes.loc[0,f'SEM_{plant}']/amplitudes.loc[0,f'Mean_{plant}'])**2 + (-2*amplitudes.loc[i,f'SEM_{plant}']/amplitudes.loc[i,f'Mean_{plant}'])**2
        sepropagated = np.sqrt(sepropagated)*gradient
        SEM_prop_list.append(sepropagated)
        
    amplitudes[f'(C0/C)^2_{plant}'] = gradient_list
    amplitudes[f'SEM_prop_{plant}'] = SEM_prop_list
    
#Plot mean diffusion plots
fig, ax = plt.subplots(figsize=(4.5,3.5),layout='tight')
ax.set_xlabel('Time, s')
ax.set_ylabel('(C$_{0,0}$/C$_{0,t}$)$^2$')
x=amplitudes['Time, s'][:60]
colors=['r','grey','k']

for i,col in enumerate(D_coeffs.columns):

    y=list(amplitudes[f'(C0/C)^2_{col}'])[:60]
    # ax.fill_between(x,y-amplitudes[f'SEM_prop_{col}'][:60],y+amplitudes[f'SEM_prop_{col}'][:60],alpha = 0.2,color = colors[i])
    # ax.errorbar(x,y,yerr=amplitudes[f'SEM_prop_{col}'], ecolor='k', linestyle='',capsize=1,fmt='o',mec='k',markerfacecolor=colors[i],label = col)
    ax.scatter(x,y,color=colors[i],edgecolor='k',label = col)
    
    popt,pcov = curve_fit(linear, x, y)
    # ax.plot(x, linear(np.array(x),*popt),color='k')
    mean_value = np.mean(D_coeffs[col])
    formatted_mean = f'{mean_value:.2e}'
    coefficient_part = formatted_mean.split('e')[0] #Use this to avoid printing the exponent part
    normalised_mean = mean_value*10**-(int(formatted_mean.split('e')[1]))
    
    std_err = np.std(D_coeffs[col], ddof=1) / np.sqrt(np.size(D_coeffs[col]))
    normalised_error = std_err*10**-(int(formatted_mean.split('e')[1]))
    
    ax.text(0.05,0.9-0.08*i,f"D$_{{{col}}}$={normalised_mean:.2f}±{normalised_error:.2f} 10$^{{{formatted_mean.split('e')[1]}}}$ cm$^2$ s$^{{{-1}}}$", transform=ax.transAxes)
ax.legend(frameon=False, loc='lower right')        
fig.savefig('Mean diffusion plot.svg')
plt.show()


