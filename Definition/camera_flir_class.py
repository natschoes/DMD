from pyspin import PySpin
import numpy as np
import time 

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
    """
    def set_exposure_time(self, exposure_time_microsec):
        if self.cam.ExposureAuto.GetAccessMode() == PySpin.RW:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        exposure_time_to_set = min(self.cam.ExposureTime.GetMax(), exposure_time_microsec)
        self.cam.ExposureTime.SetValue(exposure_time_to_set)
    """

    def setup_acquisition_mode(cam, nodemap):   #New
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if PySpin.IsWritable(node_acquisition_mode):
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if PySpin.IsReadable(node_acquisition_mode_continuous):
                acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
                node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
                print('Acquisition mode set to continuous.')
            else:
                print('Unable to set acquisition mode to continuous (entry retrieval).')
                return False
        else:
            print('Unable to set acquisition mode to continuous (enum retrieval).')
            return False
        return True

    def set_buffer_handling_mode(self):
        nodemap = self.cam.GetTLStreamNodeMap()
        
        # Get the enumeration node for buffer handling
        buffer_handling_mode = PySpin.CEnumerationPtr(nodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsReadable(buffer_handling_mode) or not PySpin.IsWritable(buffer_handling_mode):
            print('Unable to access the StreamBufferHandlingMode node.')
            return False

        # Get the entry node for "NewestOnly" mode
        newest_only_entry = buffer_handling_mode.GetEntryByName('NewestFirst')#NewestOnly
        if not PySpin.IsReadable(newest_only_entry):
            print('Unable to access the NewestOnly entry.')
            return False

        # Set the buffer handling mode to "NewestOnly"
        buffer_handling_mode.SetIntValue(newest_only_entry.GetValue())
        print('Buffer handling mode set to NewestOnly.')
        return True


    def set_exposure_time(self, exposure_time):
        try:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # Turn off auto exposure
            self.cam.ExposureTime.SetValue(exposure_time)
        except PySpin.SpinnakerException as e:
            print(f"Failed to set exposure time: {str(e)}")


    def start_acquisition(self):
        modus = 1
        try:
            if modus == 1:
            
                self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
                self.cam.BeginAcquisition()
                self.is_acquiring = True
            elif modus == 2:
                self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
                self.cam.BeginAcquisition()
                self.is_acquiring = True
        except PySpin.SpinnakerException as e:
            print(f"Failed to start acquisition: {str(e)}")


        #NEW! Preallocated list and array at the end
    def acquire_images(self, num_images, averaging=True):
        time_for_loops = []
        images = [None] * num_images  # Preallocated list with None entries
        for i in range(num_images):
            try:
                startTime = time.time()

                image_result = self.cam.GetNextImage(1000)
                endTime = time.time()

                if image_result.IsIncomplete():
                    print(f"Image incomplete with image status {image_result.GetImageStatus()}...")
                else:
                    # Directly assign the numpy array to the preallocated index
                    images[i] = image_result.GetNDArray()
                image_result.Release()
            except PySpin.SpinnakerException as e:
                print(f"Error retrieving image: {str(e)}")
                images[i] = None  # Assign None to indicate failed acquisition at this index
            deltaT = endTime - startTime
            time_for_loops.append(deltaT)
            print("One small loop took:", deltaT, "seconds", end='\r')
        # Filter out None entries and convert the list to a NumPy array outside the loop
        images = [img for img in images if img is not None]

        if images:
            images = np.array(images)
            if averaging:
                images = np.mean(images, axis=0).round().astype(np.uint8)
                print("averaged the image!")
        
        
        #deltaTime = endTime - startTime
        #print("Time for one acquire image:", deltaTime, " and has shape:", np.shape(images))
        return images, time_for_loops



    def acquire_imagesOLD(self, num_images):
        images = []
        for i in range(num_images):
            try:
                image_result = self.cam.GetNextImage()
                if image_result.IsIncomplete():
                    print(f"Image incomplete with image status {image_result.GetImageStatus()}...")
                else:
                    images.append(image_result.GetNDArray())
                    images = np.array(images)
                image_result.Release()
            except PySpin.SpinnakerException as e:
                print(f"Error retrieving image: {str(e)}")
        return images

    def stop_acquisition(self):
        try:
            if self.is_acquiring and self.cam.IsStreaming():
                self.cam.EndAcquisition()
                self.is_acquiring = False
        except PySpin.SpinnakerException as e:
            print(f"Failed to stop acquisition: {str(e)}")

    def stop_device(self):
        try:
            if self.is_acquiring and self.cam.IsStreaming():
                self.stop_acquisition()
            if self.cam.IsInitialized():
                self.cam.DeInit()
        except PySpin.SpinnakerException as ex:
            print('Error stopping device:', ex)
        finally:
            self.cam_list.Clear()
            self.system.ReleaseInstance()

    def set_gain(self, gain_value):
        """
        Set the gain of the camera to a specific value.
        
        Parameters:
        gain_value (float): The gain value to set.
        """
        if self.cam.GainAuto.GetAccessMode() == PySpin.RW:
            self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        if self.cam.Gain.GetAccessMode() == PySpin.RW:
            gain_to_set = min(self.cam.Gain.GetMax(), gain_value)
            self.cam.Gain.SetValue(gain_to_set)
            #print(f"Gain set to {gain_to_set}")
        else:
            print("Unable to set gain.")


    def __del__(self):
        self.stop_device()
