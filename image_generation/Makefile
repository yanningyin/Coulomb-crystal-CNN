default: PaulTrapSim
all: PaulTrapSim

# Detect Miniconda path and active environment dynamically
CONDA_PREFIX := $(shell conda info --base)
CONDA_ENV := $(shell conda info --envs | grep '*' | awk '{print $$1}')
INCLUDE_PATH := $(CONDA_PREFIX)/envs/$(CONDA_ENV)/include
LIB_PATH := $(CONDA_PREFIX)/envs/$(CONDA_ENV)/lib

PaulTrapSim: 
	/usr/bin/g++ -I$(INCLUDE_PATH) PaulTrapSim.cpp -L$(LIB_PATH) -lOpenMM -o PaulTrapSim.exe \
		`pkg-config --libs --cflags gtk+-2.0 cairo` -std=c++11 -Wno-deprecated-declarations
