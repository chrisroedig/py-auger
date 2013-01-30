#!/usr/bin/python

# basic system stuff
import sys
#ability to handle process signals
import signal
# cmd line options parsing
from optparse import OptionParser
# yaml parsing for config files
import yaml
# fast array manip
import numpy as np
# scientific computing
import scipy as sp
# plotting
from matplotlib import pyplot as plt
# physical constants
from scipy.constants import codata

#set pyplot to interactive mode
plt.ion()

#place holder for analyzer plugin
run_ana = None
##-----------------------------------------------------------------------------
# Initialization and Options Parsing

def init(config=None):
	print('---------------------------')
	print('\n\nPyStreak Simulation....\n')

	if(config):
		print('loading settings from '+config)
		cfg = yaml.load(file(config))
	else:
		print('loading sdefault settings')
		cfg =  yaml.load(file('default.yaml'))

	# instantiate ananlyzer
	if( 'analyzer' in cfg):
		# user specified analyzer
		ana = __import__(cfg['analyzer'])
	else:
		# default analyzer (does nothing)
		ana = __import__("ana_base")
	
	global run_ana
	run_ana = ana.analyzer()


	return cfg

##-----------------------------------------------------------------------------
# Paramer Setup

def build_shot_params(c):
	print('Setting up shot parameters')
	#set up the shot list

	if('scan' in c):
		print('Timing is in scanning mode')
		#one element for each scan step
		scan_range=np.arange(
			c['scan']['delay_start'],
			c['scan']['delay_end'],
			c['scan']['delay_step']
		)
		#scan_range --> [1,2,3,4,5]
		scan_steps = np.shape( scan_range )[0]
		event_count = scan_steps * c['scan']['shots_per_step']
		scan_array = np.array( [scan_range] * c['scan']['shots_per_step'] )

	else:
		print('Timing is in jitter mode')
		# make 
		event_count = c['shots']
		scan_array = np.zeros(event_count)


	#scan_array --> [ [1,2,3,4,5], [1,2,3,4,5],..., [1,2,3,4,5] ]
	scan_array = np.transpose(scan_array)
	#scan_array -->[ [1,1,1,1,1], [2,2,2,2,2],...,[5,5,5,5,5] ]
	scan_array = np.reshape(scan_array, (1, event_count ) )[0]
	#scan_array -->[ 1,1,1,1,1,2,2,2,2,2...,5,5,5,5,5]

	#scan_array = scan_array * (1e-15)

	# TIMING JITTER
	#gaussian for fel/ir jitter
	timing_jitter_array = np.random.normal(
		0.0, 
		c['jitter']['timing'], 
		event_count
		)

	photon_jitter_array = np.random.normal(
		c['xfel_pulse']['central_photon_energy'],
		c['jitter']['photon_energy'],
		event_count
		)

	# IR LASER STABILITY
	up_energy_array =np.random.normal(
		1.0,
		c['jitter']['ir_up']/c['ir_pulse']['pondermotive'],
		event_count
		)

	#combine all dsitributions in master shot parameter array
	shot_params={
		'time_delay': scan_array,
		'time_jitter' : timing_jitter_array,
		'photon_jitter' : photon_jitter_array,
		'ir_up' : up_energy_array,
		}
	return shot_params

def build_hit_params(c):
	# Primary ionization time (xfel pulse width)
	print('setting up hit randomization callbacks')

	def primary_time_offest(n):
		return np.random.normal( 
			0.0, 
			c['xfel_pulse']['main_pulse_duration'], 
			n
		)

	# XFEL ENERGY BANDWIDTH
	def primary_energy_offset(n): 
		return np.random.normal(
			0.0,
			c['xfel_pulse']['photon_energy_bandwidth'], 
			n
		)

	#	
	def auger_time_offest(n):  
		return np.abs(
			np.random.laplace(
				0.0, 
				c['atom']['auger_decay_time'], 
				n
				)
			)

	# AUGER ENERGY LINEWDITH
	def auger_energy(n): 
		return np.random.normal(
			c['atom']['auger_energy'],
			c['atom']['auger_energy_width'],
			n
		)

	#lambda expressions to extract hit randomization
	hit_params = {
		'primary_time_offset' : primary_time_offest,
		'primary_energy_offset' : primary_energy_offset,
		'auger_time_offest' : auger_time_offest,
		'auger_energy' : auger_energy,
	}

	return hit_params
##-----------------------------------------------------------------------------
# "Physics"

def streak(E_i,Up,omeg,Th,t):
	return E_i + 2 * Up * ( np.sin( omeg * t )**2 ) * np.cos( 2 * Th ) +\
	np.sqrt( 8 * E_i * Up ) * np.sin( omeg * t ) * np.cos( Th )

streak = np.vectorize( streak )

#function to model time evolution of IR Laser Up	
def ir_up_env(t,d,up):
	cw = ( d )/( 2 * np.sqrt( 2 * np.log(2) ) ) #implement FHWM
	return up * ( np.exp( -1*( (t)**2 ) / (2*(cw**2)) ) )**2 #gauss

ir_up_env = np.vectorize(ir_up_env)

##-----------------------------------------------------------------------------
# Main Simulation Routine

def sim(config, shots, hits):
	"""
	expected parameters:

	shots = {
		'time_delay' : <numpy array>,
		'ir_up' : <numpy array>,
		'photon_jitter' : <numpy array>,
	}
	hits = {
		'primary_time_offest' : <function>,
		'primary_energy_offset' : <function>,
		'auger_time_offest' : <function>,
		'auger_energy' : <function>,
	}
	"""
	speed_c = codata.physical_constants[ 'speed of light in vacuum'][0]
	#this is needed in rad/fs
	omeg = 2 * np.pi * speed_c / ( config['ir_pulse']['wavelength'] * 1e-6 ) * (1e-15)

	primary_spec	=	np.array([])
	auger_spec		=	np.array([])
	run_delays 				= np.array([])
	run_photons				= np.array([])

	#prepare the analyzer
	run_ana.beforeRun( config )

	print("starting run")
	print("\n\n -------------------------------")
	for i in range( 0 , len(shots['time_delay']) ):
		# for each time step, create some ionization events

		# the following params come from a pre-randomized array
		# the reflect vars that stay fixed within one shot

		# current ir/fel time offset
		time = shots['time_delay'][i] + shots['time_jitter'][i]
		run_delays = np.append( run_delays, time )

		# current ir amplitude (scale of 1.0)
		up = shots['ir_up'][i]
		
		# the central photon enegry of the xfel pulse
		photon = shots['photon_jitter'][i]
		run_photons = np.append( run_photons, photon )

		# make the params that fluctuate with each hit
		# run the distribution functions passed in as "hits"
		# the argument is the number of values to generate
		# these statements create and operate on arrays

		p_time = time+hits['primary_time_offset']( config['atom']['primary_hits'] )

		
		p_energy = \
				hits['primary_energy_offset']( config['atom']['primary_hits'] ) - \
				config['atom']['primary_binding_energy']+ \
				photon
		
		p_up = up * ir_up_env( 
				p_time, \
				config['ir_pulse']['duration'], \
				config['ir_pulse']['pondermotive']\
			)

		primary_electrons = streak( p_energy, p_up,omeg, 0.0, time )
		# primary_elecrons is the array of 
		# electron energies "detected" in this shot

		# time is ir/fel delay + random val from decay distr.
		a_time = \
			time + \
			hits['primary_time_offset']( config['atom']['auger_hits'] ) + \
			hits['auger_time_offest']( config['atom']['auger_hits'] )

		# energy is rand val within line width
		a_energy = hits['auger_energy']( config['atom']['auger_hits'] )

		# up during each auger emission
		a_up = up * ir_up_env( 
				a_time, 
				config['ir_pulse']['duration'], 
				config['ir_pulse']['pondermotive'] 
			)

		#final energy of each auger electron

		auger_electrons = streak( a_energy, a_up, omeg, 0.0, a_time )

		# these are shot fixed enviroment varibales for the analyzer
		event_env={
			'photon' : photon,# "measured" photon energy
			'time_delay' : shots['time_delay'][i], #set time delay
			'time_jitter' : shots['time_jitter'][i] #"measured" jitter position
		}
		# send event electrons and env variables to analyzer
		run_ana.event(primary_electrons,auger_electrons,event_env)

		# cumulative electron collections
		primary_spec 	= np.concatenate( (primary_spec , primary_electrons ) )
		auger_spec 		= np.concatenate( (auger_spec 	,	auger_electrons 	) )

		if( (i%100)==0 ):
			print_status(100.0*i/len(shots['time_delay']))

		if( (i%1000)==0 ):
			plot_prog( primary_spec , auger_spec,run_delays,run_photons )
			run_ana.check()
			
		
		#END for each shot
	#signal run completion to analyzer
	run_ana.afterRun()
	wait(' run complete!')
	
	return
##-----------------------------------------------------------------------------
def plot_prog( primary,	auger, delays, photons):
	primary_hist 	=	np.histogram( primary, 500)
	auger_hist		=	np.histogram( auger	,500)
	delay_hist 		=	np.histogram( delays	,50)
	photons_hist 		=	np.histogram( photons	,50)
	shot_count = len( photons )
	hit_count = len( primary )+len( auger )
	
	plt.figure(1)
	plt.clf()
	plt.title('Simulation progress')

	plt.subplot2grid((2,2),(0,0))
	plt.plot( primary_hist[1][:-1],primary_hist[0] ,lw=2)
	plt.title("Primary Spectrum (Cumulative)")

	plt.subplot2grid((2,2),(0,1))
	plt.plot( auger_hist[1][:-1], auger_hist[0] ,lw=2)
	plt.title("Auger Spectrum (Cumulative)")

	plt.subplot2grid((2,2),(1,0))
	plt.plot( delay_hist[1][:-1], delay_hist[0] )
	plt.title("IR/FEL Delay")

	plt.subplot2grid((2,2),(1,1))
	plt.plot( photons_hist[1][:-1], photons_hist[0] )
	plt.title( "Central Photon Energy" )
	plt.draw()

##-----------------------------------------------------------------------------
# helpers

#print without newline.....prettier progress update
def print_status(percent):

	status = r"Running: %d Percent | " % (percent)
	sys.stdout.write('\r' + status)
	sys.stdout.flush() # important

# wait function
def wait(str=None, prompt='Press return to continue...(this will close all plot windows)\n'):
    if str is not None:
        print str
    return raw_input(prompt)

#sigINT hander (when you hit ctrl-c)
def signal_handler(signal, frame):
	print '\n\n-----------------STOP-----------------------'
	      
	r=wait('You interrupted the simulation','Are you sure you want to quit? (y/n)')

	if(r!='y'):
		print '\n\n-----------------RESUME-----------------------'
		return
	else:
		sys.exit(0)
	return
signal.signal(signal.SIGINT, signal_handler)

##-----------------------------------------------------------------------------
##-----------------------------------------------------------------------------
##this happens on load
if __name__ == "__main__":
	parser = OptionParser()
	parser.add_option("-a", "--analyzer", dest="analyzer",
	                  help="point the simulation at your analyzer", metavar="ANALYZER")

	parser.add_option("-s", "--settings", dest="settings",
	                  help="use these settings for the simulation", metavar="SETTINGS")

	(options, args) = parser.parse_args()
	if(options.settings):
		config = init( options.settings)
	else:
		config = init()
	shots = build_shot_params( config )
	hits = build_hit_params( config )
	hist = sim( config, shots, hits )










