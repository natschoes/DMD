from pyspin import PySpin
import numpy as np
import time 
import os
#if this file is executed direclty I need this import, otherwise the next availiable one
#from camera_flir_class import FLIRCamera
from Definition.camera_flir_class import FLIRCamera
from PIL import Image

'''
def record_image(exposure, gain, image_shape, num_images, loop_index):
    camera = FLIRCamera()
    camera.set_exposure_time(exposure)
    camera.set_gain(gain)
    print("Recording image ...")
    camera.start_acquisition()

    destination_dir = r"D:\dmd thesis\fluctuation\pos_average"
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    average_image, time_loop = camera.acquire_images(num_images, image_shape, averaging=True)

    camera.stop_acquisition()
    camera.stop_device()

    if average_image is not None:
        # Save averaged image with a unique name
        file_path = os.path.join(destination_dir, f"{loop_index}.bmp")
        img_recorded = Image.fromarray(np.uint8(average_image), mode="L")
        img_recorded.save(file_path)
    else:
        print("No images were acquired successfully.")

    return time_loop

'''
#use this for com 
def record_image(exposure, gain, image_shape, num_images):
    camera = FLIRCamera()
    camera.set_exposure_time(exposure)
    camera.set_gain(gain)
    print("Recording image ...")
    camera.start_acquisition()

    destination_dir = r"D:\dmd thesis\FS testing\torrus\comp"
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    time_loop = camera.acquire_images(num_images, image_shape, destination_dir)

    camera.stop_acquisition()
    camera.stop_device()

    return time_loop

'''
def record_image(exposure, gain):
    # Second scenario: Acquiring 50 averaged images (each averaged over 10 frames)
    camera = FLIRCamera()
    camera.set_exposure_time(exposure)
    camera.set_gain(gain)
    print("Recording image ...")
    camera.start_acquisition()
    images = []
    zeiten = []
    for n in range(600):
        image, zeit = camera.acquire_images(1, averaging=True)
        images.append(image)
        zeiten.append(zeit)
    camera.stop_acquisition()
    return images, zeiten 
'''

'''
images = record_image(10,0)
image = images[0]
i=0
img_recorded = Image.fromarray(np.uint8(np.array(image)), mode="L")
#img_recorded.show()
img_recorded.save(f"saved_images/{i}test.bmp")


images = record_image()

for i in images:
    img = Image.fromarray(np.uint8((np.array(i))), mode="L")  # Convert numpy array to image
    img.show()  # Display the image
'''
