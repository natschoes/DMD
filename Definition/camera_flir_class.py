import numpy as np
import os
import time
from pyspin import PySpin
from PIL import Image

class FLIRCamera:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        if len(self.cam_list) == 0:
            self.system.ReleaseInstance()
            raise ValueError("No cameras detected.")
        self.cam = self.cam_list.GetByIndex(0)
        self.cam.Init()
        self.is_acquiring = False

    def set_exposure_time(self, exposure_time_microsec):
        if self.cam.ExposureAuto.GetAccessMode() == PySpin.RW:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        exposure_time_to_set = min(self.cam.ExposureTime.GetMax(), exposure_time_microsec)
        self.cam.ExposureTime.SetValue(exposure_time_to_set)

    def set_gain(self, gain):
        if self.cam.GainAuto.GetAccessMode() == PySpin.RW:
            self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        gain_to_set = min(self.cam.Gain.GetMax(), gain)
        self.cam.Gain.SetValue(gain_to_set)

    def start_acquisition(self):
        self.cam.BeginAcquisition()
        self.is_acquiring = True

    def stop_acquisition(self):
        self.cam.EndAcquisition()
        self.is_acquiring = False

    def stop_device(self):
        self.cam.DeInit()
        self.cam_list.Clear()
        self.system.ReleaseInstance()

    def acquire_images(self, num_images, image_shape, destination_dir):
        time_for_loops = []
        for i in range(num_images):
            try:
                start_time = time.time()
                image_result = self.cam.GetNextImage(1000)
                end_time = time.time()

                if image_result.IsIncomplete():
                    print(f"Image incomplete with image status {image_result.GetImageStatus()}...")
                else:
                    img_array = image_result.GetNDArray()
                    # Save individual image
                    file_path = os.path.join(destination_dir, f"{i}.bmp")
                    img_recorded = Image.fromarray(np.uint8(img_array), mode="L")
                    img_recorded.save(file_path)

                image_result.Release()
            except PySpin.SpinnakerException as e:
                print(f"Error retrieving image: {str(e)}")

            delta_t = end_time - start_time
            time_for_loops.append(delta_t)
            print(f"One small loop took: {delta_t} seconds", end='\r')

        return time_for_loops




    def __del__(self):
        self.stop_device()

