import datetime
import cv2
import sys
import numpy as np
import numpy.typing as npt
import time,board,busio
import adafruit_mlx90640
import matplotlib.image
import RPi.GPIO as GPIO
from libcamera import controls

from PIL import Image
from scipy.ndimage import center_of_mass
from picamera2 import Picamera2
from apriltag import apriltag

# pwm control for motors
PWMA = None # left motor
PWMB = None # right motor
# motor driver pins
PWMA_PIN = 12 # GPIO 12
AIN2_PIN = 15 # GPIO 15
AIN1_PIN = 18 # GPIO 18
STBY_PIN = 23 # GPIO 23
BIN1_PIN = 24 # GPIO 24
BIN2_PIN = 25 # GPIO 25
PWMB_PIN = 13 # GPIO 13

# TODO uncomment for Julianne testing 
i2c: busio.I2C = None
mlx: adafruit_mlx90640.MLX90640 = None
# i2c = None
# mlx = None

isdebug: bool = "-d" in sys.argv or "--debug" in sys.argv
islogdata: bool = "-ld" in sys.argv or "--log-data" in sys.argv
print_body_temp: bool = "--body-temp" in sys.argv

# thermal camera constants
THERMAL_WIDTH = 32
THERMAL_HEIGHT = 24

THERMAL_MIN = -40 # °C
THERMAL_MAX = 100 # °C

BODY_TEMP_C = 35.2
BODY_TEMP_TOLERANCE = 5

frame_height = None
frame_width = None

# gobal vars for saving video and log data
logdata_prefix = f"{datetime.datetime.now().isoformat()}".replace(":", "-")

# width,height is the two frames concatted together one scaled
video_file = logdata_prefix + ".avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video = None

# headers for logdata
logdata = [[
    "time_ms",
    "visual_on",
    "thermal_on",
    "distance_v",
    "heading_angle_v",
    "heading_angle_t",
    "centroid_v_x",
    "centroid_v_y",
    "centroid_t_x",
    "centroid_t_y",
]]

# pin for controlling program execution
RUN_PIN = 22 # GPIO 22

# pins for toggling thermal or visual sensing
THERMAL_ON = not "--no-thermal" in sys.argv
VISUAL_ON = not "--no-visual" in sys.argv

def is_run_enabled() -> bool:
    """Read pin to determine if the control loop should be ran or halted."""
    res = GPIO.input(RUN_PIN)
    return res == 1

def save_frame_to_video(frame, width, height):
    """Same `frame` to globally-stored video for the session"""
    global video
    fps = 2
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if video is None:
        video = cv2.VideoWriter(video_file, fourcc, fps, (width, height))

    if video.isOpened():
        video.write(frame)
        print("[INFO] wrote frame")
    else:
        raise RuntimeError("videowriter not opened!")

def setup_motor_driver():
    """Set up the motor driver."""
    global PWMA, PWMB
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PWMA_PIN, GPIO.OUT)
    GPIO.setup(AIN2_PIN, GPIO.OUT)
    GPIO.setup(AIN1_PIN, GPIO.OUT)
    GPIO.setup(STBY_PIN, GPIO.OUT)
    GPIO.setup(BIN1_PIN, GPIO.OUT)
    GPIO.setup(BIN2_PIN, GPIO.OUT)
    GPIO.setup(PWMB_PIN, GPIO.OUT)
    PWMA = GPIO.PWM(PWMA_PIN, 1000)
    PWMB = GPIO.PWM(PWMB_PIN, 1000)
    GPIO.output(STBY_PIN, GPIO.HIGH)
    PWMA.start(0)
    PWMB.start(0)
    # note: these must be pulled down externally
    GPIO.setup(RUN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def shut_off_motors():
    GPIO.output(STBY_PIN, GPIO.LOW)

def drive_right_motor(speed: int, direction: str):
    global PWMA
    if speed >= 0 and speed <= 100:
        PWMA.ChangeDutyCycle(speed)

    if direction == "forward":
        GPIO.output(AIN1_PIN, GPIO.LOW)
        GPIO.output(AIN2_PIN, GPIO.HIGH)
    elif direction == "backward":
        GPIO.output(AIN1_PIN, GPIO.HIGH)
        GPIO.output(AIN2_PIN, GPIO.LOW)
    else:
        raise ValueError("did not get 'forward' or 'backward'")

def drive_left_motor(speed: int, direction: str):
    global PWMB
    if speed >= 0 and speed <= 100:
        PWMB.ChangeDutyCycle(speed)

    if direction == "forward":
        GPIO.output(BIN1_PIN, GPIO.HIGH)
        GPIO.output(BIN2_PIN, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(BIN1_PIN, GPIO.LOW)
        GPIO.output(BIN2_PIN, GPIO.HIGH)
    else:
        raise ValueError("did not get 'forward' or 'backward'")

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
    """Set up the Raspberry Pi rpicam."""
    picam2 = Picamera2()
    config = picam2.create_video_configuration()
    picam2.configure(config)
    picam2.start()

    # allow picam to start up...
    time.sleep(1)

    # Get the default frame width and height
    global frame_width, frame_height
    frame_width = config["main"]["size"][0]
    frame_height = config["main"]["size"][1]

    if isdebug:
        print(f"frame_width, frame_height: {frame_width}, {frame_height}")
        print(f"exposure: {picam2.camera_controls['ExposureTime']}")
        print(f"brightness: {picam2.camera_controls['Brightness']}")

    return picam2

def process_thermal_frame(buf: npt.ArrayLike):
    global mlx, thermal_img
    try:
        mlx.getFrame(buf)
    except ValueError:
        # apparently we're supposed to ignore these
        return
    mat = np.reshape(buf, (THERMAL_HEIGHT, THERMAL_WIDTH))
    # flip the image across left-right so that its view is authentic POV
    mat = np.fliplr(mat)
    return mat

def process_april_tags(r, frame_width, frame_height):
    # get coordinates of corners
    lb = r['lb-rb-rt-lt'][0]
    rb = r['lb-rb-rt-lt'][1]
    rt = r['lb-rb-rt-lt'][2]
    lt = r['lb-rb-rt-lt'][3]

    centroid =(int(r['center'][1]), int(r['center'][0])) # the centroid reads out x as y in camera frame
    image_shape = (frame_height, frame_width) # 1080x1920 in pixel coordaintes 
    camera_fov_y = 62.2

    # NOT ROTATION INVARIANT YET
    april_tag_length_pixels = np.linalg.norm(lt - rb)

    distance, heading_angle = calculate_heading_angle(centroid, image_shape, camera_fov_y, april_tag_length_pixels)

    if isdebug:
        print(f"[INFO] April tag is at {(int(r['center'][1]), int(r['center'][0]))}")
        print("[INFO] Pixel  length:", april_tag_length_pixels)
        print(f"[INFO] The robot needs to turn {heading_angle:.2f} degrees to face the centroid.")
        print(f"[INFO] Distance {distance:.2f} in m.")

    return distance, heading_angle

def find_centroid_of_hot_area(temperature_array, threshold, tolerance):
    """
    Finds the centroid of the hot area in a temperature array.
    """
    # binary mask for the hot areas (areas above the threshold)
    hot_mask = (temperature_array - tolerance > threshold)
    if np.all(hot_mask == False):
        return (None, None)
    # find centroid using the center_of_mass function from scipy
    centroid = center_of_mass(hot_mask)
    if isdebug:
        print(f"[INFO] thermal centroid: {(int(centroid[0]), int(centroid[1]))}")
    return (int(centroid[0]), int(centroid[1]))

def calculate_heading_angle(centroid, image_shape, camera_fov_y, april_tag_length_pixels=None):
    """
    Calculate the heading angle to turn the robot towards the centroid in the image.
    
    Parameters:
    - centroid: (cx, cy) - the centroid coordinates of the hot area.
    - image_shape: (height, width) - the dimensions of the image (height, width).
    - camera_fov_x: float - the horizontal field of view of the camera in degrees.
    
    Returns:
    - heading_angle: float - the angle (in degrees) the robot needs to turn to face the centroid.
    """

    # NOTE REMEMBER THAT X IS UP DOWN AND Y IS SIDE IN CAMERA SPACE
    
    # center of the image (cx_center, cy_center)
    cx_center, cy_center = image_shape[0] // 2, image_shape[1] // 2  # image shape: (height, width)
    
    # displacement from the center to the centroid
    delta_y = centroid[1] - cy_center  # horizontal displacement in pixels
    
    # angle per pixel
    angle_per_pixel = camera_fov_y / image_shape[1]  # field of view in degrees per pixel
    
    # heading angle in degrees (relative to the center of the image)
    heading_angle = delta_y * angle_per_pixel

    # calculate depth if April tag is detected

    if april_tag_length_pixels != None:
        focal_length = frame_width 
        tag_size = 3/100 #3 cm
        distance = (tag_size * focal_length) / april_tag_length_pixels  # distance in meters
    else:
        distance = None # thermal sensing
    
    return distance, heading_angle

def end_of_sequence():
    """Run tasks for the end of the sequence."""
    global video
    shut_off_motors()
    video.release()
    video = None
    if islogdata:
        with open(logdata_prefix + ".csv", "w") as f:
            for logitem in logdata:
                f.write(",".join(map(str, logitem)))
                f.write("\n")
        print(f"[INFO] saved logdata to {logdata_prefix}.csv")

def control_loop():
    ## 0: SETUP SENSORS ##
    setup_motor_driver()
    # Visual Camera
    global frame_height, frame_width
    picam2 = setup_visual_camera()

    # Thermal camera
    global i2c, mlx, isdebug
    cam_t = setup_thermal_camera() # FIX ME how to return data?
        
    # TODO sync loop to refresh rate of slowest sensor

    distance_v = None
    heading_angle_v = None
    heading_angle_t = None
    centroid_v: tuple = (None, None)
    centroid_t: tuple = (None, None)
    visual_frame = None
    thermal_frame = None
    
    while True: 
        ## 1: DATA PROCESSING FROM SENSORS ##
        # Visual Camera
        visual_frame = picam2.capture_array("main")
        if VISUAL_ON:
            visual_greyscale = cv2.cvtColor(visual_frame, cv2.COLOR_BGR2GRAY)

            if isdebug:
                print("[INFO] detecting AprilTags...")

            detector = apriltag("tag36h11")
            results = detector.detect(visual_greyscale) # This returns the number of april tags detected

            if len(results) == 0:
                heading_angle_v = None
                distance_v = None
            else:
                # TODO - for now assume one april tag
                print("[INFO] April tag detected")
                r = results[0]
                centroid_v = (r["center"][0], r["center"][1])
                distance_v, heading_angle_v = process_april_tags(r, frame_width, frame_height)

            if isdebug:
                print("[INFO] {} total AprilTags detected".format(len(results)))
                for r in results:
                    # extract the bounding box (x, y)-coordinates for the AprilTag
                    # and convert each of the (x, y)-coordinate pairs to integers
                    lb = r['lb-rb-rt-lt'][0].astype(int)
                    rb = r['lb-rb-rt-lt'][1].astype(int)
                    rt = r['lb-rb-rt-lt'][2].astype(int)
                    lt = r['lb-rb-rt-lt'][3].astype(int)

                    # draw the bounding box of the AprilTag detection
                    cv2.line(visual_frame, lt, rt, (0, 255, 0), 2)
                    cv2.line(visual_frame, rt, rb, (0, 255, 0), 2)
                    cv2.line(visual_frame, rb, lb, (0, 255, 0), 2)
                    cv2.line(visual_frame, lb, lt, (0, 255, 0), 2)
                    # draw the center (x, y)-coordinates of the AprilTag
                    (cX, cY) = (int(r['center'][0]), int(r['center'][1]))
                    cv2.circle(visual_frame, (cX, cY), 5, (0, 0, 255), -1)
                    # draw the tag family on the image
                    # tagFamily = r.tag_family.decode("utf-8")
                    # cv2.putText(visual_frame, tagFamily, (ptA[0], ptA[1] - 15),
                    #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # print("[INFO] tag family: {}".format(tagFamily))
                    # show the output image after AprilTag detection
            
        # Thermal Camera
        thermal_buf = np.array([0] * 768)
        thermal_frame_temp = process_thermal_frame(thermal_buf)
        thermal_frame = thermal_frame_temp if thermal_frame_temp is not None else thermal_frame

        if THERMAL_ON:
            threshold_temperature = 40
            tolerance = BODY_TEMP_TOLERANCE
            # TODO might need to debug compatibility of thermal frame - 
            centroid_t = find_centroid_of_hot_area(thermal_frame, threshold_temperature, tolerance)
            
            image_shape = (THERMAL_HEIGHT, THERMAL_WIDTH)
            camera_fov_y = 110  # Horizontal FOV of the camera 

            # only calculate heading angle if centroid was found
            if centroid_t[0] is None or centroid_t[1] is None:
                heading_angle_t = None
            else:
                # Calculate the heading angle
                _, heading_angle_t = calculate_heading_angle(centroid_t, image_shape, camera_fov_y)
                print(f"[THERMAL] The robot needs to turn {heading_angle_t:.2f} degrees to face the centroid.")

        # scale the image to show regardless of if we're finding the centroid
        if isdebug:
            thermal_frame = ((thermal_frame - THERMAL_MIN) / THERMAL_MAX * 255).astype(int)
            thermal_frame = np.array(thermal_frame, dtype=np.uint8)
            # create scale to give visual and thermal images same width
            scale = int(frame_width / thermal_frame.shape[1])
            thermal_frame = cv2.resize(
                    thermal_frame,
                    (thermal_frame.shape[1] * scale, thermal_frame.shape[0] * scale),
                    interpolation=cv2.INTER_NEAREST,
            )
            thermal_frame = np.stack(
                    [thermal_frame] * 4,
                    axis = -1,
            )
            # if we're finding the centroid, display it
            if THERMAL_ON:
                # add the centroid to the scaled image
                if centroid_t[0] is not None and centroid_t[1] is not None:
                    centroid_t = (centroid_t[0] * scale, centroid_t[1] * scale)
                    cv2.circle(thermal_frame, (centroid_t[1], centroid_t[0]), 5, (0, 0, 255), -1)

        if isdebug:
            im_to_show = np.vstack((thermal_frame, visual_frame))
            save_frame_to_video(
                frame=im_to_show,
                width=im_to_show.shape[1],
                height=im_to_show.shape[0],
            )
            cv2.imshow("image", im_to_show)

        ## 2: ACTUATE MOTORS BASED ON SENSOR DATA ##
        # TODO turn towards heading angles of thermal camera and vision camera

        use_thermal: bool = THERMAL_ON and heading_angle_t is not None
        use_visual: bool = VISUAL_ON and heading_angle_v is not None

        if islogdata:
            logdata.append([
                time.time(),
                VISUAL_ON,
                THERMAL_ON,
                distance_v,
                heading_angle_v,
                heading_angle_t,
                centroid_v[0],
                centroid_v[1],
                centroid_t[0],
                centroid_t[1],
            ])

        if use_thermal and use_visual:
            avg_turn_angle = (heading_angle_v + heading_angle_t) / 2
        elif use_visual:
            avg_turn_angle = heading_angle_v 
        elif use_thermal:
            avg_turn_angle = heading_angle_t
        else:
            avg_turn_angle = None

        print(f"[INFO] average turning angle: {avg_turn_angle}")

        if avg_turn_angle is None:
            drive_right_motor(0, "forward")
            drive_left_motor(0, "forward")
        else:
            # duty cycle value is twice the turn angle
            # and gets clipped in [0,100]
            # duty_cycle_value = max(0, min(100, int(abs(avg_turn_angle * 2))))

            if avg_turn_angle > 1: # turn right
                if isdebug:
                    print("[INFO] turn right")
                drive_right_motor(10, "backward")
                drive_left_motor(10, "forward")
            elif avg_turn_angle < -1: # turn left
                if isdebug:
                    print("[INFO] turn left")
                drive_right_motor(10, "forward")
                drive_left_motor(10, "backward")
            else:
                drive_right_motor(100, "forward")
                drive_left_motor(100, "forward")



        ## 3: BREAK CONDIITON ##
        if cv2.waitKey(1) == ord('q'):
            # if 'q' is pressed, kill it.
            end_of_sequence()
            GPIO.cleanup()
            cv2.destroyAllWindows()
            exit(0)
        
        if not is_run_enabled():
            # if the RUN_PIN is disconnected, wait until it is reconnected to
            # start again.
            end_of_sequence()

            while not is_run_enabled():
                if cv2.waitKey(1) == ord('q'):
                    # if 'q' is pressed, kill it.
                    GPIO.cleanup()
                    cv2.destroyAllWindows()
                    exit(0)
                time.sleep(1)

            # rename logdata prefix to be new time
            global logdata_prefix
            logdata_prefix = f"{datetime.datetime.now().isoformat()}".replace(":", "-")
        

if __name__ == "__main__":
    control_loop()
