 # DMD - creating tailorable optical traps

This project provides code to control a Digital Micromirror Device (DMD) and to interface with a FLIR camera and. The project includes image acquisition, processing, and displaying patterns on the DMD in order to create different optical traps for the dysprosium quantum gas experiment.


## Project Sturcture
The project consists of the main file **DMD.py** which is the main file. In the folder **Definition** all necessary functions, constants and classes to run **DMD.py** are defined. The functions are defined in **Functions.py**, the constants in **Constants.py** and the class to access and interfer with the FLIR camera is instantiated in **camera_flir_class.py**


## Requirements

To run this project, following Python packages are needed
-   `numpy`
-   `scipy`
-   `matplotlib`
-   `PIL` (Pillow)
-   `PySpin` (provided by FLIR)
-   `cv2` (OpenCV)
-   `ALP4` (for DMD control, please refer to the specific DMD documentation for installation)
- `time`
-  `sys`
- `os`
- `cv2`.

To be able to install PySpin properly the [Spinnaker SDK](https://www.flir.eu/products/spinnaker-sdk/) from Teledyne FLIR has to be installed. Both, the SDK for windows as well as the SDK for python are needed. It is crucial to select the SDK for the right python version and have the latest versions of both SDKs. Otherwise there will be issues.
During the installation process of the SDK for windows one will be asked whether one wants the SDK for just SpinView or also the SDK for python including. The first option, including only SpinView has to be selected. 
After that the installation process is described in the README.txt file that comes with the download of the python SDK.

*Note:* It might be possible encountering issues trying to import the `PySpin` module. For this project importing it from `pyspin`
`from pyspin install PySpin`, 
which is unequal to `PySpin`, fixed the issue. `pyspin` can simply be installed using pip.


