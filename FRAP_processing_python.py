"""
Created on Mon Feb 12 09:56:47 2024

@author: sacco004

The script takes sequences of FRAP images, sums fluorescence intensities over one axis and 
calculates diffusion coefficients from the gaussian profiles of fluorescence bleach over time.

"""

import cv2
import numpy as np
import easygui as egui
import os
import pandas as pd
from scipy.signal import savgol_filter as sg_filter
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#Specify style to use for graphs
plt.style.use('seaborn-v0_8')

# Ensure seaborn is set up correctly
sns.set_theme(style='ticks',context='talk', palette='colorblind')
sns.despine()

def file_finder(directory,requirement):
    '''Searches in a directory and its subdirectories for csv files with a specific requirement in their name.
    directory: main directory with files and subdirectories
    requirement: string required in the file name (lowercase)'''
    
    data_files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(".tif") and (requirement in filename.lower()):
                data_files.append(os.path.join(dirpath, filename))
    return data_files

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    ''' Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)'''
        
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def profile_extractor(file):
    '''Extract the intensity profile along the y-axis of a TIFF image. 
    Returns the Image number (image name), intensity profile and length of the axis of the intensity profile.
    Example of image name to use as input: FRAP2_WT_6-41um_002.tif (ExperimentName_SampleName_ImageHeight_ImageNumber)'''
                                                                    
    name_image = file.split('\\')[-1].split('_')[3].rstrip('.tif')
    
    image = cv2.imread(file)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert the grayscale image to float64
    gray_image = np.asarray(gray_image, dtype=np.float64)
    
    #Calculate sum of intensities over y-axis
    profile = gray_image.mean(1)
        
    #Obtain x-axis (in um) for each profile
    x_axis_length = file.split('\\')[-1].split('_')[2].rstrip('um').split('-') #micro-m
    x_axis_length = float(f"{x_axis_length[0]}.{x_axis_length[1]}")
    
    return name_image, profile, x_axis_length

def Gaussian(x,C,R,u,y0):
    '''Gaussian equation to fit
    C: amplitude
    R: width
    u: centre
    y0: offset'''
    
    return y0+C*np.exp(-2*(x-u)**2/R**2)

def linear(x,m,c):
    '''Linear equation to fit
    m: gradient
    c: offset'''
    
    return m*x+c

def get_stderror(coefficients,covariance):
    '''Get standard error of the fit'''
    error = []
    for ii in range(len(coefficients)):
        try:
            error.append(np.absolute(covariance[ii][ii])**0.5)
        except:
            error.append(0.)
    return error

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

#Import files and set main directory
folder_all= egui.diropenbox("Select Folder to Import TIFF Images of a stack")

os.chdir(f'{folder_all}')
try: 
    os.chdir(f'{folder_all}\\Analysis_figures')
except: 
    os.mkdir(f'{folder_all}\\Analysis_figures')
    os.chdir(f'{folder_all}\\Analysis_figures')
    
files = file_finder(folder_all, 'frap') 

#Give user choice on whether to calculate and display fit plots (slow)
display_images = egui.buttonbox("Would you like to plot of Gaussian fits and parameters?", choices=["Yes","No"])

#Obtain list of sample names and plant names. Example of file name: PATH\FRAP10_2WT_6-91um_001.tif
samples = set([file.split('\\')[-1].rsplit('_',2)[0].rstrip('.tif') for file in files if 'leaves' not in file]) #e.g. FRAP10_2WT
plants = list(set([plant.split('_')[1]for plant in samples])) #Plants measured, including replicates
unique_samples = list(set([string[1:] if string[0].isdigit() else string for string in plants])) #Unique plant names (excluding replicates)

#Obtain time axis (1 min every 1 s) + (5 min every 5 s)
time_axis=list(np.arange(1,61,1)) + list(np.arange(61,360,5))

#Initialise dictionary of dictionaries containing image stacks
stacks = {sample:{} for sample in samples} #Stores all stack profiles
subtracted_stacks = {sample:{} for sample in samples} #Stores Post-Bleach stack profiles with Pre-Bleach subtracted
x_axis ={sample:[] for sample in samples} #Stores values of images length in um

#Import images from different samples and extract their profiles
for sample in samples:
    for file in files:
        if sample == file.split('\\')[-1].rsplit('_',2)[0].rstrip('.tif'):
            name_image, profile, x_axis_length = profile_extractor(file)
            
            # Add it to the stack
            stacks[sample][name_image] = profile
            
            # Add it to the stack
            stacks[sample][name_image] = profile
            x_axis[sample] = np.linspace(0, x_axis_length, len(profile))


    #Average pre-bleach images (5)
    pre_bleach_images = [[stacks[sample][x] for x in stacks[sample] if int(x.split('_')[-1]) in np.arange(1,6)]]
    pre_bleach_images = np.concatenate(pre_bleach_images,axis=1).T
    average_pre_bleach = pre_bleach_images.mean(1)
    
    #Delete original prebleach images (keep only the averages)
    keys_to_delete = [x for x in stacks[sample] if int(x.split('_')[-1]) in np.arange(1, 6)]
    
    for key in keys_to_delete:
        del stacks[sample][key]

    # Initialise dataframe for storage of subtracted stack profiles
    subtracted_stacks[sample] = pd.DataFrame({r"Distance, um": x_axis[sample],'Mean_pre_bleach': average_pre_bleach})
    
    for stack_key, stack_value in stacks[sample].items():
        column_suffix = int(stack_key) # Number of PB image
        
        #DataFrame storing pre-bleach mean profile and post-bleach profile 'stack_value', smoothed
        per_norm = pd.DataFrame({'Mean_pre_bleach': sg_filter(average_pre_bleach, 5, 3),f"PB_{column_suffix}":sg_filter(stack_value, 5, 3)})
        
        #Calculate 98th percentiles for normalisation
        quantile_pre = per_norm['Mean_pre_bleach'].quantile(0.98)
        quantile_post = per_norm[f"PB_{column_suffix}"].quantile(0.98)
        
        #Get subtracted stack after smoothing and normalisation
        subtracted_stack = pd.DataFrame({f"PB_{column_suffix}": per_norm['Mean_pre_bleach']/quantile_pre - per_norm[f"PB_{column_suffix}"]/quantile_post})
        subtracted_stacks[sample] = pd.concat([subtracted_stacks[sample],subtracted_stack],axis=1)

#Initialise dataframes for values of amplitude (C), radius (R) and area below Gaussian fit
amplitudes=pd.DataFrame(time_axis,columns=['Time, s'])
radii=pd.DataFrame(time_axis,columns=['Time, s'])
integrated_area=pd.DataFrame(time_axis,columns=['Time, s'])

#Calculate amplitudes of Gaussians over time

#Params for progress bar
iteration = 0
total = sum([len(subtracted_stacks[sample].columns[1:]) for sample in samples])

for sample in samples:
    
    a=[]
    r=[]
    int_A=[]
        
    #Determine initial center of the bleach (PB_6, the first image after laser bleach)
    p0= [1,1,3.5,0] # Initial guess of C,R,u,y0
    bleach_position, pcov = curve_fit(Gaussian, subtracted_stacks[sample]['Distance, um'], 
                                      subtracted_stacks[sample]['PB_6'], p0=p0,
                                      bounds=((0,0,0,-np.inf),(np.inf,np.inf,np.inf,np.inf)))
    
    #Use the bleach position as guide to fit all the next post-bleach profiles
    for stack in subtracted_stacks[sample].columns[2:]:
        
        if iteration == 0: 
            begin = time.time() #start of the countdown for progress bar
        
        #IDEA 1: Perform fit to Gaussian. Used a lambda function to fix the position of the bleach (u) 
        #from the initial bleach position.
        # p0= [1,1,0] # Initial guess of C,R,u
        # popt, pcov = curve_fit(lambda x,C,R,y0: Gaussian(x,C,R,bleach_position[2],y0), subtracted_stacks[sample]['Distance, um'], subtracted_stacks[sample][stack], p0=p0,bounds=((0,0,-np.inf),(np.inf,np.inf,np.inf)))
                
        #IDEA 2: Perform fit to Gaussian. Estimates and bounds a range of values for peak position based
        #on the initial bleach position.
        p0= [1,1,bleach_position[2],0] # Initial guess of C,R,u,yo
        try:
            popt, pcov = curve_fit(Gaussian, subtracted_stacks[sample]['Distance, um'], subtracted_stacks[sample][stack], 
                                   p0=p0, bounds=((0,0,bleach_position[2]-1,-np.inf),(np.inf,np.inf,bleach_position[2]+1,np.inf)))
        except:
            popt, pcov = np.array([np.nan]*4),np.array([[np.nan]*4]*4)
            
        # Get the standard errors of the parameters
        err = get_stderror(popt,pcov)
        
        #If the error is larger than 20% of the parameter, discard the measurment and append NaN, otherwise store a, r and int_A
        if (err[0] < 0.2*popt[0]) or (err[1] < 0.2*popt[1]):        
            a.append(popt[0])
            r.append(popt[1])
            #Simpson integration to get are below curve
            int_A.append(simpson(Gaussian(subtracted_stacks[sample]['Distance, um'],*popt)- min(Gaussian(subtracted_stacks[sample]['Distance, um'],*popt)), #y
                                 subtracted_stacks[sample]['Distance, um'])) #x
        else: 
            a.append(np.nan)
            r.append(np.nan)   
            int_A.append(np.nan)
    
        # Generate x values for the fitted curve
        x_fit = np.linspace(min(subtracted_stacks[sample]['Distance, um']), 
                            max(subtracted_stacks[sample]['Distance, um']), 1000)
        
        # Plot a few time points (slows down the code)
        try: 
            os.chdir(f'{sample}_GaussianProfiles')
        except: 
            os.mkdir(f'{sample}_GaussianProfiles')
            os.chdir(f'{sample}_GaussianProfiles')
        
        if display_images =='Yes':
            #Plot only 1 every 5 profiles
            if int(stack.split("_")[-1]) % 5 == 0:
                
                fig,ax = plt.subplots(figsize=(4,3.5),layout='tight')
                fig.suptitle(sample)
    
                # for IDEA 1:
                # ax.plot(subtracted_stacks[sample]['Distance, um'],Gaussian(subtracted_stacks[sample]['Distance, um'],*[*popt[:2],bleach_position[2],popt[-1]]),color='k')
                
                ax.plot(subtracted_stacks[sample]['Distance, um'],Gaussian(subtracted_stacks[sample]['Distance, um'],*popt),color='k')
                ax.scatter(subtracted_stacks[sample]['Distance, um'],subtracted_stacks[sample][stack],label = stack,color='w',edgecolor='k')
                ax.set_xlabel('Distance, ${\mu}$m')
                ax.set_ylabel('Intensity, A.U.')
                if (err[0] >= 0.2*popt[0]) or (err[1] >= 0.2*popt[1]): 
                    ax.text(0.5,0.5,"DISCARDED")
                ax.legend(edgecolor='k')
                fig.savefig(f"{sample}_{stack}_GaussianFit.svg")
                plt.show()
            
        os.chdir(f'{folder_all}\\Analysis_figures')
        
        #Progress bar parameters
        if iteration == 0: 
            end = time.time() 
        remaining_time = (end-begin)*(total-iteration)
        printProgressBar(iteration, total-1, prefix = 'Calculating Gaussian fitting parameters:', suffix = f'Time remaining: {remaining_time:.1f} s', length = 50)
        iteration+=1
     
    #Update the DataFrames for amplitude, radius and area
    amplitudes[sample]=a
    radii[sample]=r
    integrated_area[sample]=int_A
    
#Calculate means of amplitudes, radii and areas for each unique plant
for s in unique_samples:
    amplitudes[f'Mean_{s}']= amplitudes[[col for col in amplitudes.columns[1:] if 
                                         (s in col.split('_')[1]) and ('FRAP' in col.split('_')[0])]].mean(axis=1)
    amplitudes[f'SEM_{s}']= amplitudes[[col for col in amplitudes.columns[1:] if 
                                        (s in col.split('_')[1]) and ('FRAP' in col.split('_')[0])]].sem(axis=1)
    radii[f'Mean_{s}']= radii[[col for col in radii.columns[1:] if 
                                         (s in col.split('_')[1]) and ('FRAP' in col.split('_')[0])]].mean(axis=1)
    radii[f'SEM_{s}']= radii[[col for col in radii.columns[1:] if 
                                        (s in col.split('_')[1]) and ('FRAP' in col.split('_')[0])]].sem(axis=1)
    integrated_area[f'Mean_{s}']= integrated_area[[col for col in integrated_area.columns[1:] if 
                                         (s in col.split('_')[1]) and ('FRAP' in col.split('_')[0])]].mean(axis=1)
    integrated_area[f'SEM_{s}']= integrated_area[[col for col in integrated_area.columns[1:] if 
                                        (s in col.split('_')[1]) and ('FRAP' in col.split('_')[0])]].sem(axis=1)
      
#Plot amplitudes and radii
if display_images =='Yes':
    for sample in samples:
        
        fig,ax1 = plt.subplots(1,2,figsize=(9,3.5),layout='tight')
        # fig.suptitle(sample)
        ax1[0].scatter(amplitudes['Time, s'],amplitudes[sample],color='blue',label='C')
        ax1[0].set_xlabel('Time, s')
        ax1[0].set_ylabel('C',color='blue')
        ax1[0].tick_params(axis ='y', labelcolor = 'blue') 
        
        ax2 = ax1[0].twinx() 
        ax2.scatter(radii['Time, s'],radii[sample],color='red',label='R')
        ax2.set_ylabel('R',color='red')
        ax2.tick_params(axis ='y', labelcolor = 'red') 
        # fig.legend(loc=(0.75,0.75),frameon=False)
        
        ax1[1].scatter(amplitudes['Time, s'],integrated_area[sample],color='k',label='C')
        ax1[1].set_xlabel('Time, s')
        ax1[1].set_ylabel('Area',color='k')
        ax1[1].set_ylim(0)
        ax1[1].tick_params(axis ='y', labelcolor = 'k') 
        
        plt.savefig(f'C&R_{sample}.svg')
        
        plt.show()
    
#Plot average C and R
    colors=dict(zip(['WT','fad7fad8'],['m','g']))
    for plant in ['WT','fad7fad8']:
        
        fig,ax1 = plt.subplots(1,2,figsize=(9,3.5),layout='tight')
        
        ax1[0].scatter(amplitudes['Time, s'],amplitudes[f'Mean_{plant}'],color=adjust_lightness(colors[plant],0.6),label='C',edgecolor='k')
        ax1[0].set_xlabel('Time, s')
        ax1[0].set_ylabel('C',color=adjust_lightness(colors[plant],0.6))
        ax1[0].tick_params(axis ='y', labelcolor = adjust_lightness(colors[plant],0.6)) 
        ax1[0].set_ylim(0,1)
        
        ax2 = ax1[0].twinx() 
        ax2.scatter(radii['Time, s'],radii[f'Mean_{plant}'],color=adjust_lightness(colors[plant],1.2),
                    label='R',edgecolor='k',marker='v')
        ax2.set_ylabel('R',color=adjust_lightness(colors[plant],1.2))
        ax2.tick_params(axis ='y', labelcolor = adjust_lightness(colors[plant],1.2)) 
        ax2.set_ylim(0,1.3)
        
        ax1[1].scatter(amplitudes['Time, s'],integrated_area[f'Mean_{plant}'],color='k',label='C')
        ax1[1].set_xlabel('Time, s')
        ax1[1].set_ylabel('Area',color='k')
        ax1[1].set_ylim(0,1.2)
        ax1[1].tick_params(axis ='y', labelcolor = 'k') 
        
        plt.savefig(f'C&R_{plant}_mean.svg')
        plt.show()
        

#Calculate and plot diffusion coefficients. Calculated over the first minute of post-bleach

#Initialise DataFrame where to store diffusion coefficients
D_coeffs=pd.DataFrame(columns=unique_samples)

# Create an empty dictionary to store Series
D_coeffs_dict = {}

for plant in unique_samples:
    D=[]
    fig, ax = plt.subplots(figsize=(4.5,3.5),layout='tight')
    fig.suptitle(f'{plant}')
    ax.set_xlabel('Time, s')
    ax.set_ylabel('(C$_{0,0}$/C$_{0,t}$)$^2$')
    
    for sample in samples:
        if (plant in sample.split('_')[1]) and ('FRAP' in sample.split('_')[0]):
            
            x=amplitudes['Time, s'][:60]
            y=list((amplitudes[sample][0]/amplitudes[sample])**2)
            amplitudes[f'(C0/C)^2_{sample}'] = y
            
            ax.scatter(x,amplitudes[f'(C0/C)^2_{sample}'][:60])
            popt,pcov = curve_fit(linear, x, amplitudes[f'(C0/C)^2_{sample}'][:60])
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

        D_coeffs_dict[plant] = pd.Series(D)
    
    # Create a DataFrame from the dictionary
    D_coeffs = pd.DataFrame(D_coeffs_dict)
    
    ax.text(0.1,0.8,f"{np.mean(D):.2E} cm$^2$ s$^{-1}$", transform=ax.transAxes)
        
    fig.savefig(f'Diffusion plot_{plant}.svg')
    plt.show()
    
#Initialise DataFrame where to store diffusion coefficients
D_coeffs_after_10s=pd.DataFrame(columns=unique_samples)

# Create an empty dictionary to store Series
D_coeffs_dict_after_10s = {}

for plant in unique_samples:
    D=[]
    fig, ax = plt.subplots(figsize=(4.5,3.5),layout='tight')
    fig.suptitle(f'{plant}')
    ax.set_xlabel('Time, s')
    ax.set_ylabel('(C$_{0,0}$/C$_{0,t}$)$^2$')
    
    for sample in samples:
        if (plant in sample.split('_')[1]) and ('FRAP' in sample.split('_')[0]):
            
            x=amplitudes['Time, s'][10:60]
            y=list((amplitudes[sample][0]/amplitudes[sample])**2)
            amplitudes[f'(C0/C)^2_{sample}'] = y
            
            ax.scatter(x,amplitudes[f'(C0/C)^2_{sample}'][10:60])
            popt,pcov = curve_fit(linear, x, amplitudes[f'(C0/C)^2_{sample}'][10:60])
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

        D_coeffs_dict_after_10s[plant] = pd.Series(D)
    
    # Create a DataFrame from the dictionary
    D_coeffs_after_10s = pd.DataFrame(D_coeffs_dict_after_10s)
    
    ax.text(0.1,0.8,f"{np.mean(D):.2E} cm$^2$ s$^{-1}$", transform=ax.transAxes)
        
    fig.savefig(f'Diffusion plot_after10s_{plant}.svg')
    plt.show()

#Calculate propagated errors

for plant in unique_samples:
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
#ax[0] : (C0/Ci)^2 vs time
#ax[1] : boxplot with diffusion coefficients

fig, ax = plt.subplots(1,2,figsize=(10,4),layout='tight')
ax[0].set_xlabel('Time, s')
ax[0].set_ylabel('(C$_{0,0}$/C$_{0,t}$)$^2$')
x=amplitudes['Time, s'][:61]
colors=dict(zip(['WT','fad7fad8'],['m','g']))

for i, col in enumerate([sample for sample in unique_samples if sample != 'npq1+DCMU']):

    y=list(amplitudes.loc[0:60,f'(C0/C)^2_{col}'])

    ax[0].fill_between(x,y-amplitudes[f'SEM_prop_{col}'][:61],
                       y+amplitudes[f'SEM_prop_{col}'][:61],
                       alpha = 0.2,color = colors[col])
    ax[0].scatter(x,y,color=colors[col],edgecolor='k',label = col,linewidth = 1)
    
    popt,pcov = curve_fit(linear, x, y)
    # ax.plot(x, linear(np.array(x),*popt),color='k')
    mean_value = np.mean(D_coeffs[col])
    formatted_mean = f'{mean_value:.2e}'
    coefficient_part = formatted_mean.split('e')[0] #Use this to avoid printing the exponent part
    normalised_mean = mean_value*10**-(int(formatted_mean.split('e')[1]))
    
    std_err = np.std(D_coeffs[col], ddof=1) / np.sqrt(np.size(D_coeffs[col]))
    normalised_error = std_err*10**-(int(formatted_mean.split('e')[1]))
    

ax[0].legend(frameon=False,loc = 'lower right')        #bbox_to_anchor=(1, 0.5)
ax[1] = sns.boxplot(data=D_coeffs[['WT','fad7fad8']]*1e11,
            palette=colors.values())
ax[1].set(ylabel='D (cm$^2$ s$^{{{-1}}}$ *10$^{{{-11}}}$)')

fig.savefig('Mean diffusion plot_60.svg')
plt.show()

#Plot mean diffusion plots with D coeffs calculated after 10 s
#ax[0] : (C0/Ci)^2 vs time
#ax[1] : boxplot with diffusion coefficients

fig, ax = plt.subplots(1,2,figsize=(10,4),layout='tight')
ax[0].set_xlabel('Time, s')
ax[0].set_ylabel('(C$_{0,0}$/C$_{0,t}$)$^2$')
x=amplitudes['Time, s'][:61]
colors=dict(zip(['WT','fad7fad8'],['m','g']))

for i, col in enumerate([sample for sample in unique_samples if sample != 'npq1+DCMU']):

    y=list(amplitudes.loc[0:60,f'(C0/C)^2_{col}'])

    ax[0].fill_between(x,y-amplitudes[f'SEM_prop_{col}'][:61],
                       y+amplitudes[f'SEM_prop_{col}'][:61],
                       alpha = 0.2,color = colors[col])
    ax[0].scatter(x,y,color=colors[col],edgecolor='k',label = col,linewidth = 1)
    
    popt,pcov = curve_fit(linear, x, y)
    # ax.plot(x, linear(np.array(x),*popt),color='k')
    mean_value = np.mean(D_coeffs_after_10s[col])
    formatted_mean = f'{mean_value:.2e}'
    coefficient_part = formatted_mean.split('e')[0] #Use this to avoid printing the exponent part
    normalised_mean = mean_value*10**-(int(formatted_mean.split('e')[1]))
    
    std_err = np.std(D_coeffs_after_10s[col], ddof=1) / np.sqrt(np.size(D_coeffs_after_10s[col]))
    normalised_error = std_err*10**-(int(formatted_mean.split('e')[1]))
    

ax[0].legend(frameon=False,loc = 'lower right')        #bbox_to_anchor=(1, 0.5)
ax[1] = sns.boxplot(data=D_coeffs_after_10s[['WT','fad7fad8']]*1e11,
            palette=colors.values())
ax[1].set(ylabel='D (cm$^2$ s$^{{{-1}}}$ *10$^{{{-11}}}$)')

fig.savefig('Mean diffusion plot_60_after10s.svg')
plt.show()

#%% PCA approach

from sklearn.decomposition import PCA as sklearnPCA

# test_sample = np.random.choice([col for col in subtracted_stacks])

df_PC1 =  pd.DataFrame(columns= [key for key,value in subtracted_stacks.items()])

for test_sample in subtracted_stacks:
    data_array = []
    samples_pca = subtracted_stacks[test_sample][[col for col in subtracted_stacks[test_sample] if 'PB' in col]]
    for xx in samples_pca:
        data_array.append([y for y,z in zip(list(samples_pca[xx]),time_axis)])
    
    x_pca = subtracted_stacks[test_sample]['Distance, um']
    
    plt.figure(layout='tight')
    plt.title(test_sample)
    plt.plot(x_pca,np.asarray(data_array).T)
    plt.ylabel('Fluorescence, A.U.')
    plt.xlabel('Distance, ${\mu}$m')
    plt.savefig(f'PCA_allProfiles_{test_sample}.svg')
    plt.show()
    
    extracted_data = np.asarray(data_array).T
    
    global_average = np.average(extracted_data,axis=1)
    aves = np.asarray([(np.average(x)) for x in extracted_data.T])
    sklearn_pca2 = sklearnPCA()
    PCA_arrays = np.asarray([(x-np.average(x)) for x in extracted_data.T]).T
    # PCA coefficients
    sklearn_transf_ch2 = sklearn_pca2.fit_transform(PCA_arrays)
    # Principal components
    python_scores_ch2 = sklearn_pca2.fit(PCA_arrays).components_
    # Varience explained per component
    varience_values_ch2 = sklearn_pca2.explained_variance_
    PC_component_number2 = np.arange(1., len(varience_values_ch2) + 0.1, 1)
    
    explained_variance2 = (varience_values_ch2/sum(varience_values_ch2))*100
    
    scree_plot, ax1 = plt.subplots(layout='tight')
    ax1.plot(PC_component_number2,explained_variance2, 'ko--')
    ax1.set_ylabel('Total Variance Explained (%)')
    ax1.set_xlabel('Principal Component')
    scree_plot.savefig(f'PCA_screePlot_{test_sample}.svg')
    
    comp_1 = python_scores_ch2[0]
    comp_2 = python_scores_ch2[1]
    PC1 = -comp_1
    
    adjusted_weights_PC1 = (-sklearn_transf_ch2[:,0])*max((-comp_1-min(-comp_1)))
    adjusted_weights_PC2 = (-sklearn_transf_ch2[:,0])*max((-comp_2-min(-comp_2)))

    #Plot PCA decomposition result for PC1
    figure1,[ax1,ax2] = plt.subplots(1,2, figsize=(12,6),layout='tight')

    ax1.plot(x_pca,global_average,'k-',label='Average FRAP profile')
    ax1.plot(x_pca,adjusted_weights_PC1,'b-',label='PC 1 ('+'{:.1f}'.format(explained_variance2[0])+' % variance explained)')
    ax1.set_xlabel('Distance, ${\mu}$m')
    ax1.set_ylabel('Average Intensity')
    ax1.legend(edgecolor='k',loc='upper center', bbox_to_anchor=(0.5, 1.4),
          ncol=1, fancybox=True, shadow=True)
    ax2.plot(time_axis,PC1,'bo')
    ax2.set_ylabel('PC 1 Coefficients')
    ax2.set_xlabel('Post-bleach time (min)')
    # ax2.set_ylim((1.5*min(PC1),1.5*max(PC1)))
    figure1.savefig(f'PCA_decomposition_{test_sample}.svg')
    plt.show()
    
    #Plot PC1*adjusted weigths
    fig, ax = plt.subplots(layout='tight')
    ax.set_ylabel('Fluorescence, A.U.')
    ax.set_xlabel('Distance, ${\mu}$m')
    for coeff in PC1:
        ax.plot(x_pca,coeff*adjusted_weights_PC1)
    plt.savefig(f'PCA_reconstructedPC1_{test_sample}.svg')
    plt.show()
    
    #Plot PC2*adjusted weigths
    fig, ax = plt.subplots(layout='tight')
    ax.set_ylabel('Fluorescence, A.U.')
    ax.set_xlabel('Distance, ${\mu}$m')
    for coeff in -comp_2:
        ax.plot(x_pca,coeff*adjusted_weights_PC2)
    plt.savefig(f'PCA_reconstructedPC2_{test_sample}.svg')
    plt.show()
    
    df_PC1[test_sample] = PC1
    
    # fig, ax = plt.subplots()
    # ax.scatter(time_axis,[(PC1[0]/coeff)**2 for coeff in PC1])
    # plt.show()

#Final diffusion plot for WT and fad calculated from PC1

fig, ax = plt.subplots(figsize=(5,3.8),layout='tight')
ax.set_xlabel('Time, s')
ax.set_ylabel('(C$_{0,0}$/C$_{0,t}$)$^2$')
ax.set_xlim(-2,62)
ax.set_ylim(0.75,2.75)
colors={'WT':'m','fad':'g'}

for samp in ['WT','fad']:

    y = df_PC1[[x for x in df_PC1.columns if (samp in x)]].mean(axis=1)
    err = df_PC1[[x for x in df_PC1.columns if (samp in x)]].sem(axis=1)
    
    gradient = np.array([(y[0]/coeff)**2 for coeff in y]) #(c0/Ci)**2
    
    se_PC = (2*err[0]/y[0])**2 + (-2*err/y)**2
    se_PC = np.sqrt(se_PC) * gradient
    
    ax.scatter(time_axis,gradient,c=colors[samp],edgecolor='k',linewidth = 1)
    ax.fill_between(time_axis,gradient-se_PC,gradient+se_PC,alpha = 0.2,color=colors[samp])
    
fig.savefig('PCA_PC1_diff_plot.svg')
plt.show()

#%% Simulate time series of bleach profiles

# Create simulation of gaussian with varying C and R

x_sim = np.linspace(0, 6,1000)
y_sims = []
for C,R in zip(np.linspace(0.8,0.4,120),np.linspace(0.7,1.1,120)):
    y_sim = Gaussian(x_sim, C, R, 3, 0)
    y_sims.append(y_sim)
    
data_array = []
samples_pca = y_sims
for xx in samples_pca:
    data_array.append([y for y,z in zip(xx,np.linspace(0, 6,1000))])

x_pca = np.linspace(0, 6,1000)

plt.plot(x_pca,np.asarray(data_array).T)
extracted_data = np.asarray(data_array).T

global_average = np.average(extracted_data,axis=1)
aves = np.asarray([(np.average(x)) for x in extracted_data.T])
sklearn_pca2 = sklearnPCA()
PCA_arrays = np.asarray([(x-np.average(x)) for x in extracted_data.T]).T
# PCA coefficients
sklearn_transf_ch2 = sklearn_pca2.fit_transform(PCA_arrays)
# Principal components
python_scores_ch2 = sklearn_pca2.fit(PCA_arrays).components_
# Varience explained per component
varience_values_ch2 = sklearn_pca2.explained_variance_
PC_component_number2 = np.arange(1., len(varience_values_ch2) + 0.1, 1)

explained_varience2 = (varience_values_ch2/sum(varience_values_ch2))*100

scree_plot, ax1 = plt.subplots()
ax1.plot(PC_component_number2,explained_varience2, 'ko--')
ax1.set_ylabel('Total Variance Explained (%)')
ax1.set_xlabel('Principal Component')

comp_1 = python_scores_ch2[0]
PC1 = (-comp_1-min(-comp_1))/max((-comp_1-min(-comp_1)))
PC2 = (python_scores_ch2[1]-min(python_scores_ch2[1]))/max((-python_scores_ch2[1]-min(-python_scores_ch2[1])))

adjusted_weights_PC1 = (-sklearn_transf_ch2[:,0])*max((-comp_1-min(-comp_1)))
adjusted_weights_PC2 = (sklearn_transf_ch2[:,1])*max((python_scores_ch2[1]-min(python_scores_ch2[1])))

figure1 = plt.figure(figsize=(10,4),layout='tight')
gs = GridSpec(2, 2, figure=figure1)
ax1 = figure1.add_subplot(gs[:,0])
ax2 = figure1.add_subplot(gs[0,1])
ax3 = figure1.add_subplot(gs[1,1])
figure1.suptitle(test_sample)
ax1.plot(x_pca,global_average,'k-',label='Global Average Gaussian profile')
ax1.plot(x_pca,adjusted_weights_PC1,'b-',label='PC 1 ('+'{:.1f}'.format(explained_varience2[0])+' % variance explained)')
ax1.plot(x_pca,adjusted_weights_PC2,'-',color='mediumvioletred',label='PC 2 ('+'{:.1f}'.format(explained_varience2[1])+' % variance explained)')
ax1.set_xlabel('Distance (um)')
ax1.set_ylabel('Average intensity, Normalised PCs')
ax1.legend(edgecolor='k')
ax2.plot(time_axis,PC1,'bo')
ax2.set_ylabel('PC 1 Coefficients')
ax2.set_xticklabels([])
ax3.plot(time_axis,PC2,'o',color='mediumvioletred')
ax3.set_ylabel('PC 2 Coefficients')
ax3.set_xlabel('Post-bleach time (min)')
ax2.set_ylim((1.5*min(PC1),1.5*max(PC1)))
ax3.set_ylim((1.5*min(PC1),1.5*max(PC1)))

figure1.savefig('Gaussian_simulation.svg')
