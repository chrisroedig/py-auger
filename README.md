# py-auger 
This is a framework for simulating an XFEL experiment that intends to make direct time resolved measurments of atomic auger decays or other electron emission events.
This code is a work in progress....obviously

## Experiment Concept (in pictures only)

![Experiment Concept](https://raw.github.com/chrisroedig/py-auger/master/info/auger-exp-concept.png)

## Simulation Block Diagram

![Sim Diagram](https://raw.github.com/chrisroedig/py-auger/master/info/sim-diagram.png)

## Example Results

### Comparing Various Decay Times
![Jitter Results](https://raw.github.com/chrisroedig/py-auger/master/info/sim-jitter-dtime.png)

### Comparing Theory, Scanned Simualtion and Jittered Simulation
![Reconstruction Results](https://raw.github.com/chrisroedig/py-auger/master/info/sim-comp.png)

## Requirements:

Most of the dependencies are met by installing a python environment geared towards technical computing. The pip system (python package manager) allows easy installation of these depedencies as well.

* python  >2.7
* numpy
* scipy
* matplotlib
* pyyaml


## Usage :

### direct terminal exec:

* assuming a *NIX setup here (macOS, linux etc.)
* make sure sim.py is executable `chmod +x sim.py`
* you may need to change the interpreter callout `#!/usr/bin/python` in the first line of the file `which python` will tell you where your python is
* to run the sim with all default values: `./sim.py`
* to run the sim with a desired settings file `./sim.py -s my_config.yaml`
* to interrupt a running simulation hit `ctrl-c`, you'll get a chance to resume or shut down the simulation

### module import

You can import sim.py in the python shell and operate it from there. This might be nice for wrapping automation around it, especially once data saving is implemented (in a couple days).

* import the main sim module `import sim`
* instantiate the simulation `sim_engine = py_auger_engine()`
* load your configuration `sim_engine.setup( 'configfile.yaml' )` (omit argument to load default config)
* run the simulation `sim_engine.sim()`


## Configuration

The simulation conditions are defined in the .yaml files. 
I've inlcuded examples for:

* `neon_jitter.yaml` -  Neon  ( Primary -> 870eV, Auger -> 804eV ) with 1600 eV xrays and 4.0 um IR pulses
* `carbon_jitter.yaml` - CO2  ( Primaru -> 100eV, Auger -> ~250eV ) with 450 eV xrays and 17.0 um IR pulses

the config files are conmented to explain the values.

## Analysis:

The simulation reduces data as the experiment would and feeds a limited set of event data and enviroment values to the analyzer. The idea is to test certain analysis routines.
The analyzer to use is called out in the config file. The simulation will attempt to load a module by that name. If the analyzer is not defined, it will load the analyzer base class which does nothing.
The included analyzer `ana_1` utilizes a basic sorting routine and has been shown to reproduce the decay envelope quite well (in siumlation).
Othe analyzers can be easily created as long as they inherit the analyzer base class and properly and implements all of the callbacks.



