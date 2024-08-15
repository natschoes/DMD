import numpy as np

######constants for image import
#ensure file path exists otherwise you will get an error in the main file for importing constants
#this path corresponds to the image i recorded with the camera
image_path = r"D:\dmd thesis\FS testing\torrus\final\0.bmp"

#this path corresponds to the image i-1 that was convolved with the PSF after the error was added 
img_path_old = r"C:\Users\lavoine\Desktop\Andor python\previous.bmp"

#######constants for DMD
#folder where all recorded images from the camera are stored
images_folder = r"D:\dmd thesis\FS testing\torrus\new try"
Rotation_Angle = -46.95 #-46.95
display_time = 5800
display_time_iterate = 300
mirror_state= 1
mirror_size = 10.8e-6         # = a
#size of the dmd in pixels
Nx = 1920
Ny = 1080

Demagnification=(np.sqrt(180000)*1.85)*(300*10.8)**(-1)
magnification = 291/200
ratio_px=10.8/1.85

######constants for error calculation for the FS algorithm
minP=80
Kp=0.2

#########constants for convolution of i-1 image with the PSF
sigmax=1.0
sigmay=1.0
newsize=50
size_of_resize = 200

#constants for torrus 
outer_radius = 200
inner_radius = 130

####constants to calculate the error for the plot
minP_scaled=43.6#/80*minP
num_angles = 360
num_samples = 71

###constants to save plots
save_folder = r"C:\Users\lavoine\Desktop\Andor python"
folder_path = r"D:\dmd thesis\FS testing\torrus\new try"


#constants for cropping boundaries of the recorded image #[1571:2153,1429:2011], fine [1557:2140,1411:1994] 
top_exp =1571
bottom_exp = 2153#1993#
left_exp = 1429
right_exp = 2011#1847#


#constants for cropping boundaries of the target image; for resized torrus: [669:1251,249:831]; [669:1252,249:832]
top_target = 669
bottom_target = 1251
left_target = 249
right_target = 831


#constants to crop polar transformed image [:,269:413] 272, 409 for final try
min_r = 269
max_r = 410

#####constants for the camera
exposure = 10
gain = 0


#constant for amount of iterations
number_of_iterations = 15
