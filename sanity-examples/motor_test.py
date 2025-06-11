import time
import RPi.GPIO as GPIO

# pwm control for motors
PWMA = None # left motor
PWMB = None # right motor
# motor driver pins
PWMA_PIN = 32  # GPIO 14
AIN2_PIN = 10 # GPIO 15
AIN1_PIN = 12 # GPIO 18
STBY_PIN = 16 # GPIO 23
BIN1_PIN = 18 # GPIO 24
BIN2_PIN = 22 # GPIO 25
PWMB_PIN = 33 # GPIO 8

def setup_motor_driver():
    """Set up the motor driver."""
    global PWMA, PWMB
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PWMA_PIN, GPIO.OUT)
    GPIO.setup(AIN2_PIN, GPIO.OUT)
    GPIO.setup(AIN1_PIN, GPIO.OUT)
    GPIO.setup(STBY_PIN, GPIO.OUT)
    GPIO.setup(BIN1_PIN, GPIO.OUT)
    GPIO.setup(BIN2_PIN, GPIO.OUT)
    GPIO.setup(PWMB_PIN, GPIO.OUT)
    GPIO.output(STBY_PIN, GPIO.HIGH)
    PWMA = GPIO.PWM(PWMA_PIN, 1000)
    PWMB = GPIO.PWM(PWMB_PIN, 1000)
    PWMA.start(0)
    PWMB.start(0)

def shut_off_motors():
    GPIO.output(STBY_PIN, GPIO.LOW)

def drive_left_motor(speed: int, direction: str):
    global PWMA
    if speed >= 0 and speed <= 100:
        PWMA.ChangeDutyCycle(speed)

    if direction == "forward":
        GPIO.output(AIN1_PIN, GPIO.HIGH)
        GPIO.output(AIN2_PIN, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(AIN1_PIN, GPIO.LOW)
        GPIO.output(AIN2_PIN, GPIO.HIGH)
    else:
        raise ValueError("did not get 'forward' or 'backward'")

def drive_right_motor(speed: int, direction: str):
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

setup_motor_driver()
print("[INFO] driving left motor at 100%")
drive_left_motor(100, "forward")
time.sleep(3)
print("[INFO] driving right motor at 100%")
drive_right_motor(100, "forward")
time.sleep(3)
GPIO.cleanup()
