import sys
import cv2 as cv
import numpy as np
import numpy.typing as npt
import time,board,busio
import adafruit_mlx90640
import matplotlib.image

i2c: busio.I2C = None
mlx: adafruit_mlx90640.MLX90640 = None

isdebug: bool = "-d" in sys.argv or "--debug" in sys.argv
print_body_temp: bool = "--body-temp" in sys.argv

THERMAL_WIDTH = 24
THERMAL_HEIGHT = 32

BODY_TEMP_C = 35.2
BODY_TEMP_TOLERANCE = 5


def setup_thermal_camera():
    """Set up the thermal camera via I2C."""
    global i2c, mlx
    i2c = busio.I2C(board.SCL, board.SDA, frequency=800e3)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    print("MLX address detected on I2C", [hex(i) for i in mlx.serial_number])

    # note: can also use REFRESH_16_HZ, REFRESH_8_HZ, REFRESH_4_HZ, REFRESH_2_HZ, and others
    # (see source for more)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ


def setup_visual_camera():
    """Set up the visual camera."""
    # FIXME
    pass


def print_normalized_matrix(w: int, h: int, mat: npt.ArrayLike):
    """Print out an ascii representation of a matrix."""
    min_val = mat.min()
    max_val = mat.max()
    for i in range(w):
        for j in range(h):
            val = mat[i,j]
            disp = (" .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#M&W8%B@$")[round((val - min_val) / (max_val - min_val) * 69)]
            print(disp, end=" ")
        print()
    print()


def process_thermal_frame(ind: int, buf: npt.ArrayLike):
    global mlx, thermal_img
    try:
        mlx.getFrame(buf)
    except ValueError:
        # apparently we're supposed to ignore these
        return
    mat = np.reshape(buf, (THERMAL_WIDTH, THERMAL_HEIGHT))
    if print_body_temp:
        # calculate mask of values that fall within what we predict a body temperature to be
        body_temp_mask = cv.inRange(
            mat,
            BODY_TEMP_C - BODY_TEMP_TOLERANCE,
            BODY_TEMP_C + BODY_TEMP_TOLERANCE,
        )
        print_normalized_matrix(THERMAL_WIDTH, THERMAL_HEIGHT, body_temp_mask)
    if isdebug:
        print_normalized_matrix(THERMAL_WIDTH, THERMAL_HEIGHT, mat)
        fname = f"thermal-data-{ind:03}.png"
        result = cv.imwrite(fname, mat)
        print(f"image saving result: {fname}", result)


def process_visual_frame():
    pass


def main():
    global i2c, mlx, isdebug

    setup_thermal_camera()
    setup_visual_camera()
    thermal_buf = np.array([0] * 768)
    i = 0
    while True:
        process_thermal_frame(i, thermal_buf)
        process_visual_frame()
        i += 1
    

if __name__ == "__main__":
    main()
