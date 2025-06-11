# Pit Viper-Inspired Integration of Thermal and Visual Sensing

This project contains the source code, design files, and run data for our
project that integrates thermal and visual sensing.

## Directory Structure

```sh
.
├── apriltag . . . . . . . . . apriltag source code
├── data . . . . . . . . . . . cad models and data from runs
│  ├── mechanical-data
│  └── runs
├── sanity-examples  . . . . . minimal working examples for our own sanity
│  ├── thermal-sensing
│  ├── camera_test.py
│  ├── motor_test.py
│  ├── picam2_array_test.py
│  └── picamera-test.py
├── src  . . . . . . . . . . . main source code
│  └── main.py
├── README.md
├── requirements.txt . . . . . libraries used in source code
└── setup.sh . . . . . . . . . setup script for RPi4
```

## How to run

To set up this project, first source `setup.sh` on a Raspberry Pi 4 image running
Bookworm or later.

```sh
. setup.sh
```

This will take care of initializing the venv and installing system packages.

Next, install the packages from `requirements.txt`:

```sh
pip install -r requirements.txt
```

Finally, run the main source code, using `-d` for debug mode, and `-ld` for data
logging mode. You can optionally disable either thermal or visual sensing by
using `--visual-only` and `--thermal-only` respectively. By default, both
sensing modalities are in use.

```sh
python3 src/main.py [-d] [-ld] [--visual-only] [--thermal-only]
```

## For reference

* https://www.circuitstoday.com/arduino-nano-tutorial-pinout-schematics
* https://learn.sparkfun.com/tutorials/tb6612fng-hookup-guide/all#hardware-setup
