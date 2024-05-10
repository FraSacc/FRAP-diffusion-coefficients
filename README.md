# Fluorescence recovery after photobleaching (FRAP) - Calculate diffusion coefficients
The script takes sequences of FRAP images, sums fluorescence intensities over one axis and calculates diffusion coefficients from the Gaussian profiles of fluorescence bleach over time.

## First step: importing and preparing data 
Local data are stored as time-series microscopy images of FRAP experiments. Example of image name: "~\FRAP10_2WT_6-91um_001.tif", where "FRAP10" denote the biological replicate, "2WT" is the name of the sample,"6-91um" is the length of the y-axis, "001" is the image number in the time series.
Steps:
* User selects folder to scan for TIFF images. Subfolders are also checked.
* The script opens the images with the 'OpenCV' library and integrates the fluorescence intensity over the y-axis. The result is an intensity profile for each image of the time series.
* Images ('stacks') are stored in a dictionary of dictionaries of pandas DataFrames, where the first level is the sample, the second level is the biological replicate and each dataframe contains the fluorescence profile of the time series images.
* Prebleach images are then averaged to get a smoother baseline, and subtracted to each time series image to obtain the subtracted profile. Before each calculation, profiles are smoothed with a Savitzky-Golay filter and normalised to the 98th percentile value of the fluorescence intensity signal. This step is necessary for a consistent and reliable fitting procedure in later steps.

## Gaussian fitting of subtracted fluorescence intensity profiles
The one-dimensional diffusion equation for a single fluorophore can be expressed as such: 

$$\frac{\partial C}{\partial t}=D\frac{\partial^2 C}{\partial t^2} \tag{1}$$

where $C_{(y,t)}$ is the concentration of the bleached fluorophore, t is time, y is distance and D is the diffusion coefficient. \
Assuming an initial Gaussian profile of the bleaching region, so that:

$$C_{(y,t=0)}=C_{(y=0,t=0)}*e^{-2y^2/R_0^2} \tag{2}$$

where $R_0$ is the half-width ($\frac{1}{e^2}$) of the bleach, which is centred at $y=0$, then the solution to equation (1) becomes:

$$C_{(y,t=0)}=C_{(y=0,t=0)}\frac{R_0}{\sqrt{(R_0^2+Dt)}}*e^\frac{-2y^2}{R_0^2+8Dt} \tag{3}$$
