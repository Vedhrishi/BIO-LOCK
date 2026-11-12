"""BioLock SSD - face-recognition lock for external drives.

Author: Pasupuleti Vedh Rishi
"""

import os

# Cameras opened through Media Foundation start several seconds faster without OpenCV's
# hardware transforms. This has to be set before cv2 is imported anywhere.
os.environ.setdefault('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS', '0')

__version__ = '10.0'
