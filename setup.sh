#!/usr/bin/env bash

#########################################
#########################################
#### NOTE: run using `source` or `.` ####
####                                 ####
#### Example: source setup.sh        ####
#########################################
#########################################

GREEN="\033[0;32m"
NC="\033[0m"

echo -e "${GREEN}Pulling submodules...${NC}"
git submodule update --init --recursive

echo -e "${GREEN}Installing picamera2, cmake, and build-essential...${NC}"
# note: do not install picamera2 via pip. It will not handle compatibility
# issues in the backend
sudo apt install -y python3-picamera2
# for building apriltag
sudo apt install -y cmake
sudo apt install -y build-essential

echo -e "${GREEN}Setting up apriltag...${NC}"
# set up apriltag and compile
git submodule update --init --recursive
cd apriltag
cmake -B build -DCMAKE_BUILD_TYPE=Release
# do not install apriltag system-wide. complains about perms
cmake --build build
cd ..

echo -e "${GREEN}Creating virtualenv...${NC}"
# if venv does not exist, create it
if [ ! -d "venv" ]; then
	virtualenv --system-site-packages venv
fi

# exit existing virtual envirnment if already in one
if [ -n "$VIRTUAL_ENV" ]; then
	deactivate
fi

echo -e "${GREEN}Activating virtualenv...${NC}"
. venv/bin/activate

echo -e "${GREEN}Adding apriltag to python search path...${NC}"
# append apriltag build directory to python search path
export PYTHONPATH=apriltag/build/:$PYTHONPATH
