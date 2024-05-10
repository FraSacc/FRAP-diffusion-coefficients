# Fluorescence recovery after photobleaching (FRAP) - Calculate diffusion coefficients
The script takes sequences of FRAP images, sums fluorescence intensities over one axis and calculates diffusion coefficients from the Gaussian profiles of fluorescence bleach over time.

## First step: importing and preparing data 
Local data are stored as time-series microscopy images of FRAP experiments. Example of image name: "~\FRAP10_2WT_6-91um_001.tif", where "FRAP10" denote the biological replicate, "2WT" is the name of the sample,"6-91um" is the length of the y-axis, "001" is the image number in the time series.

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

Data of each subtracted fluorescence profile was fitted to equation (2) using scipy.optimize and parameters for amplitude (C), radius (R) and total area under the fit (A) are stored in separate DataFrames.
* The first step involves estimating the initial guesses of the fit based on the parameters obtained after fitting the first bleach image (t=0). This ensures the algorithm doesn't struggle at later time points where the signal-to-noise ratio worsens significantly.
* For instances where the error is larger than 20% of the parameter, the measurement is discarded and NaN is appended.
* Plots of fitting examples (one every 5) are shown and saved.
* Means and standard errors are calculated. Average C, R and A values are plotted and saved.

## Calculate diffusion coefficients 
Based on equation (3), a plot of $(\frac{C_{(y=0,t=0)}}{C_{(y=0,t)}})^2$ over time should give a linear relationship with slope m = $\frac{8D}{R_0^2}$. Diffusion coefficients (D) can then be derived.
* A linear equation is fitted to $(\frac{C_{(y=0,t=0)}}{C_{(y=0,t)}})^2$ over time.
* Diffusion coefficients are calculated as $D = \frac{mR_0^2}{8}$.
* Propagated errors are calculated as $(2\delta C_{0,0}/\overline{C}\_{0,0})^2$ + $(2\delta C_{0,t}/\overline{C}_{0,t})^2$
* Finally, diffusion plots as well as boxplots with diffusion coefficients are displayed.

## Principal Component Analysis (PCA) alternative approach
Since the fitting procedure relies heavily on the initial guesses and microscopy images are subject to practical issues (e.g. rotation of the sample over time, drift, focus shift, etc.), another approach was explored, relying on PCA decomposition. 
* PCA is performed on the 125 intensity profiles of each single experiment, using the PCA module of sklearn.decomposition.
* A scree plot is used to determine the variance explained by each principal component. Since the majority of the variance (more than 92.5%) is explained by PC1, only that is analysed further. Plotting the weighted PC1 component at different time points, shows that it reflects the decrease in the amplitude of the bleach over time. For this reason, it can be used to draw diffusion plots.
* 



