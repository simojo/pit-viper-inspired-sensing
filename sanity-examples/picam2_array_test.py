from picamera2 import Picamera2
import time
import cv2
import numpy as np


def rpicamera2_capture(isdebug: bool) -> np.ndarray
    """Capture a greyscale image using rpicamera2"""
    picam2 = Picamera2()
    picam2.start()

    time.sleep(1)

    array = picam2.capture_array("main")

    grey = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    cv2.imshow("ajk;a;ajkl;", grey)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
