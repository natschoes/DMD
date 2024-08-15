import PIL
import numpy as np
import scipy
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import PIL.Image as Image
import time
import sys
#from PIL import image
import matplotlib as mpl
from ALP4 import *
from PIL import Image
import cv2
import os
import os
from scipy.optimize import curve_fit
import scipy
import cv2 

#Functions
def plot_intensity(image_paths,Rotation_angle):
    plt.figure()
    i=0
    for image_path in image_paths:
        img = Image.open(image_path).rotate(Rotation_angle)
        img_gray = img.convert("L")
        img_array = np.array(img_gray)
        row_intensity = np.mean(img_array, axis=1)
        plt.plot(row_intensity, label=i)
        i+=1
    plt.xlabel("Rows")
    plt.ylabel("Intensity")
    plt.title("Intensity of Rows")
    plt.legend()
    plt.show()


def display_DMD(img, nbImg, display_time):
    # Load the Vialux .dll
    DMD = ALP4(version = '4.3',libDir="./ALP-4.3 API") 
    # initialize the device
    DMD.Initialize()
    # Binary amplitude image

    # Quantization of the image between 1 (on/off) and 8 (256 pwm grayscale levels).
    bitDepth = 1
    # WARNING even for a boolean quantization, the values readed are 0 and 255.
    # Array generation

    imgAff = img[0].ravel(order='F')*255 # Modification to a 1D array for the DMD

    if nbImg>1: #Array generation for a sequence with more than 1 image
        for i in range(nbImg-1):
            np.concatenate([imgAff,(img[i+1].ravel(order='F')*255)])
    
    # Allocate the onboard memory for the image sequence
    # you can load many pictures change only nbImg
    seq1 = DMD.SeqAlloc(nbImg, bitDepth = bitDepth)

    # Send the image sequence as a 1D list/array/numpy array
    # enter the images of the sequences

    DMD.SeqPut(imgData =imgAff.ravel(order='F'))

    DMD.SeqControl(controlType=2104 , value=2106) # activation off the uninterrupted mode

    # set image rate to 50 Hz so period of 2e4 µs
    DMD.SetTiming(pictureTime = 2000) 

    # Run the sequence in an infinite loop
    DMD.Run(loop=True)

    ###there is two method for stopping the DMD: after a chosen time of after pressing enter. Only one method work when the programm is running ( not in comment )
    time.sleep(display_time) # stop the programm for the display time (in seconds) but let the DMD running his sequence

    #input("Press enter to stop the DMD") # stop the sequence display after pressing enter
    # stop the sequence display
    DMD.Halt()
    # Free the sequence from the onboard memory
    DMD.FreeSeq()
    #De-allocate the device
    DMD.Free()
    print("END of display")
    return


### Functions creating different images

def cross(NumberPixel, ligne_croix, colonne_croix, epaisseur_ligne, epaisseur_colonne):
    ##creates a cross on the DMD
    img = np.zeros([NumberPixel[0], NumberPixel[1]], dtype=int)

    # horizontal line
    img[int(ligne_croix - epaisseur_ligne/2) : int(ligne_croix + epaisseur_ligne/2), :] = 1

    # vertical line
    img[: , int(colonne_croix - epaisseur_colonne/2) : int(colonne_croix + epaisseur_colonne/2)] = 1
    return img



def uniform(NumberPixel, a):
    """ Creates a uniform image on the DMD (a = 0 or 1) """
    img = np.zeros([NumberPixel[0], NumberPixel[1]], dtype=int)
    img[:,:] = a
    return img



def k_pixels(NumberPixel, k, a, position_ligne, debut_trait):

    """
    Creates an image with k pixels equal to a (a = 0 or 1)

    line_position is the line on which the few pixels will be changed

    start_trait is the column where the stroke begins
    """

    # Line smaller than DMD

    assert debut_trait + k <= NumberPixel[1]

    # Some black pixels on a white background

    if a==0:
        img = np.ones([NumberPixel[0], NumberPixel[1]], dtype=int)
        img[position_ligne, debut_trait : debut_trait + k] = 0

    # A few white pixels on a black background

    else:
        img = np.zeros([NumberPixel[0], NumberPixel[1]], dtype=int)
        img[position_ligne, debut_trait : debut_trait + k] = 1

    return img


def point(NumberPixel):
    img = np.zeros([NumberPixel[0], NumberPixel[1]], dtype=int)
    img[960,540]=1
    return img


def points_magnification(NombrePixel):
    img = np.zeros([NombrePixel[0], NombrePixel[1]], dtype=int)
    img[960, 540] = 1
    img[961, 540] = 1
    img[960, 541] = 1
    img[961, 541] = 1



    # Create a 14x14 rectangle centered around (1256, 540)
    start_x = 1256 - 4  # 7 pixels left of 1256
    end_x = 1256 + 4  # 7 pixels right of 1256
    start_y = 540 - 4  # 7 pixels above 540
    end_y = 540 + 4  # 7 pixels below 540

    img[start_x:end_x + 1, start_y:end_y + 1] = 1

    img[960, 840] = 1
    img[961, 840] = 1
    img[961, 841] = 1
    img[960, 841] = 1
    return img



def rectangle(NumberPixel, centre, widthdividedbytwo, lengthdividedbytwo, a):

    """Creates a rectangle with the value a (a = 0 or 1) on the DMD.

    center must be of the form (center_row, center_column)

    Attention, NumberPixel = (number of column, number of row)

    """
    # Rectangle smaller than the DMD

    assert centre[1] - widthdividedbytwo >= 0

    assert centre[1] + widthdividedbytwo <= NumberPixel[1]

    assert centre[0] - lengthdividedbytwo >=0

    assert centre[0] + lengthdividedbytwo <= NumberPixel[0]

    if a == 0:
        img = np.ones([NumberPixel[0], NumberPixel[1]], dtype=int)
        for ligne in range(centre[0] - lengthdividedbytwo, centre[0] + lengthdividedbytwo):
            img[ligne, centre[1] - widthdividedbytwo : centre[1] + widthdividedbytwo ] = 0

    else :
        img = np.zeros([NumberPixel[0], NumberPixel[1]], dtype=int)
        for ligne in range(centre[0] - lengthdividedbytwo, centre[0] + lengthdividedbytwo):
            img[ligne, centre[1] - widthdividedbytwo : centre[1] + widthdividedbytwo ] = 1

    return img




def display_torus_and_rectangle(NumberPixel, outer_radius, inner_radius, rectangle_size):
    # Create a torus
    torus_img = torrus(outer_radius, inner_radius)

    # Define the rectangle parameters
    centre = (rectangle_size[0] // 2 + 500, rectangle_size[1] // 2 + 400)  # Center of rectangle
    widthdividedbytwo = rectangle_size[1] // 2
    lengthdividedbytwo = rectangle_size[0] // 2
    a = 1  # Rectangle value

    # Create a rectangle in the top-left corner
    rectangle_img = rectangle(NumberPixel, centre, widthdividedbytwo, lengthdividedbytwo, a)

    # Combine the images (together, torus + rectangle)
    combined_img = torus_img + rectangle_img
    
    # Display the combined image
    plt.imshow(combined_img, cmap='gray')
    plt.title('Torus and Rectangle in Top-Left Corner')
    plt.axis('off')
    plt.show()
    
    return combined_img

    

def circle(r):
    img = np.zeros([1920, 1080], dtype=int)
    for x in range(1920):
        for y in range(1080):
            if((x-960)**2+(y-540)**2)<(r*r):
               img[x,y]=1
    return img


def torrus(outer_radius, inner_radius):
    return (circle(outer_radius)-circle(inner_radius))>0.5


def sort_bmp_files(folder_path):        
    # List all BMP files in the directory
    bmp_files = [f for f in os.listdir(folder_path) if f.endswith('.bmp')]
    
    # Sort the files based on the integer part of the filename
    sorted_bmp_files = sorted(bmp_files, key=lambda x: int(os.path.splitext(x)[0]))
    
    return sorted_bmp_files


def plot_intensities_torrus(bmp_files, folder_path, Rotation_angle, min_r,max_r, top_exp, bottom_exp, left_exp, right_exp):
    plt.figure()
    i=0
    for bmp_file in bmp_files:
        # Load the image
        img_path = os.path.join(folder_path, bmp_file)
        image = Image.open(img_path).convert('L').rotate(Rotation_angle)  # Convert to grayscale and rotate
        img_array = np.transpose(np.array(image))[top_exp:bottom_exp,left_exp:right_exp]
 
        #--- the following holds the square root of the sum of squares of the image dimensions ---
        #--- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
        value = np.sqrt(((img_array.shape[0]/2.0)**2.0)+((img_array.shape[1]/2.0)**2.0))
        polar_image = cv2.linearPolar(img_array,(img_array.shape[0]/2, img_array.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)
        polar_image = polar_image.astype(np.uint8)[:,min_r:max_r]
        '''
        plt.figure(1)
        plt.imshow(polar_image)
        plt.colorbar()
        plt.title(f"polar transformed image for {bmp_file}")
        plt.show()

        polar_image_converted=Image.fromarray(np.uint8((np.array(polar_image))),mode="L")
        polar_image_converted.save(f"polar_image{bmp_file}") 
        '''
        row_intensity = np.mean(polar_image, axis=1)
        num_rows = polar_image.shape[0]
        angles = np.linspace(0, 2 * np.pi, num_rows)
        plt.plot(angles, row_intensity, label=bmp_file)
        i+=1
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    plt.ylabel('Average intensity')
    #plt.title(f'Average Intensity')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_from_conv(errors,iterations,ylabel):    
    # Ensure iterations and errors_final have the same length
    assert len(iterations) == len(errors), "Lengths of iterations and errors_final do not match"

    min_index = np.argmin(errors)
    max_index = np.argmax(errors)
    print(f"Minimum of {ylabel}: ", np.min(errors), f"for iteration {min_index}.", f"\nMaximum of {ylabel}: ", np.max(errors), f"for iteration {max_index}.")
    plt.scatter(iterations, errors, color="red", label='Data')
    plt.plot(iterations, errors, 'b-', label='')
    plt.xlabel('Iteration')
    plt.ylabel(f'{ylabel}')
    plt.title(f'')
    plt.legend()
    plt.show()  


def convolve_target(target_exp, sigmax, sigmay):
    newsize = target_exp.shape[0]
    x=np.linspace(-newsize//2,newsize//2,newsize)
    y=np.linspace(-newsize//2,newsize//2,newsize)
    exp_convol=np.ones((newsize,newsize))       
    for i in range(exp_convol.shape[0]):
        for k in range(exp_convol.shape[0]):
            exp_convol[i,k]=1/2/np.pi/sigmax/sigmay*np.exp(-1/2*x[i]**2/sigmax**2-1/2*y[k]**2/sigmay**2)
    target_exp=scipy.signal.fftconvolve(target_exp,exp_convol,mode="same")  #this is the image the camera should record in a theoretical and ideal world
    target_exp_cropped=target_exp#[869:1051,449:631]    #crop out just the target
    target_exp2_resized = Image.fromarray(np.uint8(target_exp_cropped), mode="L").resize((newsize, newsize))
    '''
    plt.figure(1)
    plt.imshow(target_exp2_resized)
    plt.title("")
    plt.colorbar()
    plt.show()
    '''
    return target_exp_cropped

def calculate_error_from_convolution(bmp_files, target_exp, sigmax, sigmay, folder_path, Rotation_angle, minP_scaled, inner_radius, outer_radius, magnification, top_exp, bottom_exp, left_exp, right_exp, min_r,max_r):
    print("Calculating error from convolution...")

    errors_calculated = []
    flatness_calculated = []
    mean_intensities = []
    
    convolved_target = convolve_target(target_exp,sigmax, sigmay)

    for bmp_file in bmp_files:
        # Load, rotate, transpose and crop the image
        img_path = os.path.join(folder_path, bmp_file)
        image = Image.open(img_path).convert('L').rotate(Rotation_angle)  # Convert to grayscale and rotate
        image = np.transpose(np.array(image))
        image_cropped = image[top_exp:bottom_exp,left_exp:right_exp]#[1560:2142,1416:1998]
        
        len_rows, len_cols = image_cropped.shape
        
        plt.figure(1)
        plt.imshow(image_cropped, cmap='viridis')
        plt.title("image experiment")
        plt.colorbar()

        plt.figure(2)
        plt.imshow(minP_scaled*target_exp, cmap='viridis')
        plt.title("target experiment")
        plt.colorbar()

        plt.show()
        #exit(0)
        
        ###ERROR####################
        #calculate the error and save the file to later on save pixel values in initialized lits and plot
        error =  minP_scaled*convolved_target - image_cropped

        # Get the dimensions of the image
        height, width = error.shape

        center = (width // 2, height // 2)
        inner_radius_temp = inner_radius*magnification-1
        outer_radius_temp = outer_radius*magnification+1

        # Create a grid of coordinates
        y, x = np.ogrid[:height, :width]

        # Calculate the distance from the center
        distance_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        print("inner radius: ", inner_radius_temp, "outer radius: ", outer_radius_temp)
        # Create the mask for the torus
        mask = (distance_from_center >= inner_radius_temp) & (distance_from_center <= outer_radius_temp)

        #first transform the cropped image to polar coordinates to calculate the mean intensity of the torrus
        value = np.sqrt(((image_cropped.shape[0]/2.0)**2.0)+((image_cropped.shape[1]/2.0)**2.0))
        polar_image = cv2.linearPolar(image_cropped,(image_cropped.shape[0]/2, image_cropped.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)
        image_cropped_polar = polar_image.astype(np.uint8)[:,min_r:max_r]

        len_rows, len_cols = image_cropped_polar.shape

        
        # Apply the mask to the error image, setting values outside the torus to zero
        error_cleaned = np.where(mask, error, 0)

        error=error_cleaned
        
        error_calculated = 100*(np.sqrt(1/(len_cols * len_rows)*np.sum((error/(np.max(image_cropped_polar)-np.min(image_cropped_polar)))**2)))
        errors_calculated.append(error_calculated)

        ###FLATNESS########################################

        mean_intensity = np.sqrt(np.sum(image_cropped_polar)/(len_rows * len_cols))
        print("mean intensity: ", mean_intensity)

        flatness = mean_intensity*target_exp - image_cropped

        real_mean_intensity = np.mean(image_cropped_polar)
        mean_intensities.append(real_mean_intensity)

        flatness_calc = 100*(1-np.sqrt(1/(len_cols * len_rows)*np.sum(((flatness)/mean_intensity)**2)))

        flatness_calculated.append(flatness_calc)

    return errors_calculated, flatness_calculated, mean_intensities


def plot_mean_intensity_of_each_iteration(iterations, mean_intensities):
    plt.scatter(iterations, mean_intensities, color="red", label='Data')
    plt.plot(iterations, mean_intensities, 'b-', label='')
    plt.xlabel('Iteration')
    plt.ylabel('mean intensities')
    #plt.title(f'')
    plt.legend()
    plt.show()  


def convolve_and_save_new_image(new_img, newsize, sigmax, sigmay, size_of_resize):
    #convolution of image with added error with psf
    x=np.linspace(-newsize//2,newsize//2,newsize)
    y=np.linspace(-newsize//2,newsize//2,newsize)

    exp_convol=np.ones((newsize,newsize))       
    for i in range(exp_convol.shape[0]):
        for k in range(exp_convol.shape[0]):
            exp_convol[i,k]=1/2/np.pi/sigmax/sigmay*np.exp(-1/2*x[i]**2/sigmax**2-1/2*y[k]**2/sigmay**2)
    img_conv=scipy.signal.fftconvolve(new_img,exp_convol,mode="same")
    img1=Image.fromarray(np.uint8((np.array(img_conv))),mode="L")
    img1.save("previous.bmp") 

    img2=img1.resize((size_of_resize, size_of_resize))

    newarray=np.array(img2) 
    return newarray


def create_floyd_steinberg_image(newarray,Nx,Ny):
    #RESCALING to necessary array size
    background=PIL.Image.new(mode="L",size=(Ny,Nx)) #creates array with zeros in size of dmd
    background_array=np.array(background)

    dim_img=newarray.shape
    print("Dimension convolved image: ", dim_img)
    for i in range(dim_img[0]):
        for k in range(dim_img[1]):
            background_array[(960-dim_img[1]//2)+i, (540-dim_img[0]//2)+k]+=newarray[i,k] 

    #for normalization of dithering
    #print("max background array pre setting pixel values >255=255: ", np.max(background_array))
    background_array[background_array>255]=255
    #print("max background array pre 255*np.max(): ", np.max(background_array))
    background_array=np.array((255/np.max(background_array))*background_array)
    #print("max background array: ", np.max(background_array))


    #DITHERING
    imgfinal=Image.fromarray(np.uint8(background_array),mode="L")       
    img_FS=np.array(imgfinal.convert("1",dither=Image.Dither.FLOYDSTEINBERG))#>0.5  

    return img_FS


def preprocess_recorded_image(image_path, top_exp, bottom_exp, left_exp, right_exp, Rotation_angle):
    imgbmp=Image.open(image_path).rotate(Rotation_angle)
    im_array=np.transpose(np.array(imgbmp))
    im_array_cropped=im_array[top_exp:bottom_exp,left_exp:right_exp]#[1560:2142,1416:1998]

    return im_array_cropped

def preprocess_old_image(img_path_old, Rotation_angle):
    previous_image=Image.open(img_path_old).rotate(Rotation_angle)
    previous_array=np.transpose(np.array(previous_image))

    return previous_array


def filter_background_noise_for_torrus(error, inner_radius, outer_radius, magnification):
    ###filter out errors caused by noise of the background###
    height, width = error.shape

    center = (width // 2, height // 2)
    inner_radius_1 = inner_radius*magnification
    outer_radius_1 = outer_radius*magnification

    # Create a grid of coordinates
    y, x = np.ogrid[:height, :width]

    # Calculate the distance from the center
    distance_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Create the mask for the torus
    mask = (distance_from_center >= inner_radius_1) & (distance_from_center <= outer_radius_1)

    # Apply the mask to the error image, setting values outside the torus to zero
    error_cleaned = np.where(mask, error, 0)
    error1=error_cleaned

    return error1
