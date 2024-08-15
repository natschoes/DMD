############HEADER################################################
import PIL
import numpy as np
import scipy
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import PIL.Image as Image
import time

import sys

import matplotlib as mpl

from ALP4 import *

from PIL import Image

from PIL import Image, ImageDraw

import os
from scipy.optimize import curve_fit
import scipy
import cv2


import threading
import time


#package needed to access camera. the package is distributed by FLIR. please look into README for installation guidance
from pyspin import PySpin
#from Definition.camera_flir_class import FLIRCamera

from Definition import Constants as const
from Definition import Functions as func
from Definition import access_camera as access_c



# Main code

if __name__ == "__main__":

    #######IMPORT CONSTANTS############################################ 
    #If you want to adjust them, please look into Definition\Constants.py
    
    #constants for DMD
    images_folder = const.images_folder
    display_time = const.display_time
    display_time_iterate = const.display_time_iterate
    mirror_state= const.mirror_state
    mirror_size = const.mirror_size        # = a
    Nx = const.Nx
    Ny = const.Ny
    pixel_number = (Nx, Ny)

    Demagnification=const.Demagnification
    magnification = const.magnification
    Rotation_angle=const.Rotation_Angle
    ratio_px=const.ratio_px

    #import images
    image_path = const.image_path
    img_path_old = const.img_path_old

    #constants for calculation of the error
    minP=const.minP
    Kp=const.Kp

    #constants for calculation of convolution with the PSF
    newsize=const.newsize
    sigmax=const.sigmax
    sigmay=const.sigmay
    size_of_resize=const.size_of_resize

    #constants for torrus 
    outer_radius = const.outer_radius
    inner_radius = const.inner_radius

    ###constants to save plots
    save_folder = const.save_folder
    folder_path = const.folder_path

    #constants to calculate plots
    minP_scaled=const.minP_scaled
    num_angles = const.num_angles
    num_samples = const.num_samples

    #constants for cropping boundaries of the recorded image
    top_exp = const.top_exp
    bottom_exp = const.bottom_exp
    left_exp = const.left_exp
    right_exp = const.right_exp

    #constants for cropping boundaries of the target image
    top_target = const.top_target
    bottom_target = const.bottom_target
    left_target = const.left_target
    right_target = const.right_target

    #constants to crop polar transformed image [:,269:413]
    min_r = const.min_r
    max_r = const.max_r

    #####constants for the camera
    exposure = const.exposure
    gain = const.gain
    
    #constant for amount of iterations
    number_of_iterations = const.number_of_iterations
    ######START#################################################################################

    ####Creating of the target pattern####################
    img_rect = func.rectangle(pixel_number, (960,540), 100, 100, a=mirror_state)#-rectangle(pixel_number, (960,540), largeursur2=199, longueursur2=199, a=mirror_state)
    img_points=func.points_magnification(pixel_number)
    


    for noi in range(number_of_iterations):
        list_of_sorted_images = func.sort_bmp_files(images_folder)
        #print("Length of list_of_sorted_images:", len(list_of_sorted_images))  # Debugging
        #for the display and the first iteration the image 0.bmp has to be used as a reference
        if noi>=2:
            try:
                image_path = list_of_sorted_images[noi]
                #print(f"image path for iteration {noi}: ", image_path)
            except IndexError:
                print(f"Error: No image found for iteration {noi}.")
        else: 
            image_path = list_of_sorted_images[0]

        print(f"image path for iteration {noi}: ", image_path)
        #####LOAD IMAGES, crop and rotate them accordingly#####
        #apply operations on noi'th image in the folder. So make sure always the fresh recorded image is taken in
        #some image has to be in the folder. So place one that will be overwritten after that. !!only bmp files get taken into account
        im_array_cropped = func.preprocess_recorded_image(image_path, top_exp, bottom_exp, left_exp, right_exp, Rotation_angle)
        img_exp=im_array_cropped
        #im_array_cropped=im_array[1320:1780,1175:1635] #for points magni

        img_old=Image.open(img_path_old)


        ######DEFINE REFERENCE PATTERN the error calculation is based on#########
        rect_size=im_array_cropped.shape[0]
        pixel_number_target=[rect_size,rect_size]
        print("pixel_number_target", pixel_number_target)
        center=[rect_size//2,rect_size//2] 
        #target_img=255*points_magnification(pixel_number)[900:1300, 500:800]

        target_img=minP*func.rectangle(pixel_number_target,center, rect_size//2, rect_size//2,a=mirror_state)

        #torrus_resized = func.torrus(outer_radius*magnification, inner_radius*magnification)
        #target_img = minP*torrus_resized[top_target:bottom_target,left_target:right_target]#[669:1251,249:831]

        #for display and 1st iteration img_old=target_img, then please outcomment
        if noi<2:
            img_old=target_img
            print(f"Image old is set equal to the target image, currently at iteation {noi}.")    
    
        '''
        plt.figure(1)
        plt.imshow(target_img)
        plt.title("target_img")
    
        plt.figure(2)
        plt.imshow(img_exp)
        plt.title("img_exp cropped")
    
        plt.figure(3)
        plt.imshow(previous_array)
        plt.title("previous_array")

        plt.show()
        #exit(0)
        '''


    
        ##################### PID LOOP and CREATION OF THE NEW IMAGE ################
        error=Kp*(target_img-img_exp)
    
        #use this only for torrus
        #error = func.filter_background_noise_for_torrus(error, inner_radius, outer_radius, magnification)

        new_img=img_old+error   #hier noch alles fein
        #img_close() needs to be OUTCOMMENTED for display and first iteration. Otherwise you will get an error.
        #img_old.close()
        if noi>=2:
            img_old.close()
            print(f"Image old closed.")   


        '''
        plt.figure(1)
        plt.imshow(error)
        plt.colorbar()
        plt.title("error")
        plt.show()
    
        plt.figure(2)
        plt.imshow(img_old)
        plt.colorbar()
        plt.title("old img")
        plt.show()

        plt.figure(3)
        plt.imshow(new_img)
        plt.colorbar()
        plt.title("old img with added error")
        plt.show()
        '''

        ###################### CONVOLUTION WITH THE PSF ##################
        #convolution of image with added error with psf
        newarray=func.convolve_and_save_new_image(new_img, newsize, sigmax, sigmay, size_of_resize)


        ############## CREATION OF THE NEW FLOYD STEINBERG IMAGE ####################

        img_FS=func.create_floyd_steinberg_image(newarray,Nx,Ny)


        '''
        plt.figure(0)
        plt.imshow(im_array_cropped,'Blues')
        plt.title("imported image")
        plt.colorbar()

        plt.figure(1)
        plt.imshow(img_FS,'Blues')
        plt.title("final FS image, dithered and resized convolution")
        plt.colorbar()

        plt.figure(2)
        plt.imshow(background_array,'Blues')
        plt.title("Resized background img")
        plt.colorbar()

        plt.figure(3)
        plt.imshow(new_img,'Blues')
        plt.title("Error added to old img")
        plt.colorbar()
    
        plt.show()
        '''

        #####DISPLAY ON THE DMD##############################################
        #make sure that for iteration 0 the target is displayed and otherwise the FS image
        if noi==0:
            t1=threading.Thread(target=func.display_DMD, args=[[img_rect], 1, display_time_iterate])
            print("Display target pattern")
        else: 
            t1=threading.Thread(target=func.display_DMD, args=[[img_FS], 1, display_time_iterate])
            print("Display FS image")

        t1.start()
        #access_c.record_image returns a list of images, function can also be used for averaging.
        images = access_c.record_image(exposure, gain)
        t1.join()
        image = images[0]

        img_recorded = Image.fromarray(np.uint8(np.array(image)), mode="L")
        #img_recorded.show()
        img_recorded.save(f"saved_images/{noi}.bmp")
        print(f"Image for iteration {noi} saved.")
        time.sleep(5)
        #exit(0)
        '''
        camera = FLIRCamera()
        camera.set_exposure_time(10)
        #camera.set_buffer_handling_mode()
        camera.start_acquisition()
        image_recorded = camera.acquire_images(1, averaging=False)
        camera.stop_acquisition()
        camera.stop_device()
        image_recorded.save("test.bmp")
        print("Image recorded successfully.")
        '''
        #returns 50 images
        ###########PLOT#####################################################
        #create pattern to calculate the error and the flatness
        '''
        torrus_resized = (func.circle(outer_radius*magnification)-func.circle(inner_radius*magnification))>0.5
        target = torrus_resized[top_target:bottom_target,left_target:right_target]

        bmp_files = func.sort_bmp_files(folder_path)

        errors_final, flatness_final, mean_intensities = func.calculate_error_from_convolution(bmp_files, target, sigmax, sigmay, folder_path, Rotation_angle, minP_scaled, inner_radius, outer_radius, magnification, top_exp, bottom_exp, left_exp, right_exp, min_r,max_r)

        iterations=np.arange(len(errors_final)).tolist()

        func.plot_from_conv(flatness_final,iterations,ylabel="flatness in %")
        func.plot_from_conv(errors_final,iterations,ylabel="error rms in %")

        func.plot_intensities_torrus(bmp_files, folder_path, Rotation_angle, min_r,max_r, top_exp, bottom_exp, left_exp, right_exp)
        func.plot_mean_intensity_of_each_iteration(iterations, mean_intensities)
        '''
        noi+=1
