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

####################################################################
####################################################################
## py-auger engine class
##
####################################################################


class py_auger_engine():

	#inputs
	config 			= None
	shot_params = None
	hit_params 	= None

	#analyzer
	ananlyzer 	= None

	#cumulative data histograms
	data_hist = {}

	#cumulative data binning
	data_bin = {}

	def __init__(self):
		"""
		---------------------------------------------------------------------------
		---------------------------------------------------------------------------
			CONSTRUCTOR
		"""
		print('---------------------------')
		print('\n\nPyStreak Simulation....\n')

		self.streak 		= np.vectorize( self.streak )
		self.ir_up_env 	= np.vectorize( self.ir_up_env )

		return
	
	def setup(self,configfile):
		"""
		---------------------------------------------------------------------------
		---------------------------------------------------------------------------
			PREPARE EVERYTHING
		"""
		print("PREPARING....")
		self.load_config(configfile)
		self.setup_histograms()
		self.build_shot_params()
		self.build_hit_params()
		print("SIMULATION READY")

	def setup_histograms(self):
		"""
		---------------------------------------------------------------------------
		---------------------------------------------------------------------------
			SET UP CUMULATIVE HISTOGRAMS
		"""
		dcfg = self.config['data']
		xcfg = self.config['xfel_pulse']
		jcfg = self.config['jitter']
		acfg = self.config['atom']
		icfg = self.config['ir_pulse']

		#set up the data histogram
		auger_energy = \
			( dcfg['auger_bins']  , ( dcfg['auger_min'], dcfg['auger_max']) )
		
		primary_energy = \
			(	dcfg['primary_bins'],	( dcfg['primary_min'], dcfg['primary_max']) )

		
		delay_min = -3*self.config['jitter']['timing']
		delay_max = 3*self.config['jitter']['timing']
		
		if('scan' in self.config):
			delay_max+=self.config['scan']['delay_end']
			delay_min+=self.config['scan']['delay_start']

		delay = (	100,	( delay_min, delay_max ) )		



		photon_width = np.sqrt(
			( xcfg['photon_energy_bandwidth']**2) + 
			( jcfg['photon_energy']**2)
		)
		photon = (
				100,
				(
					xcfg['central_photon_energy'] - 2 * photon_width,
					xcfg['central_photon_energy'] + 2 * photon_width
				)
			)

		primary_angle = (
				100,
				( 0, acfg['primary_full_angle'] )
			)

		auger_angle = (
				100,
				( 0, acfg['auger_full_angle'] )
			)
		
		up = (
				100,
				( 
					0,
					icfg['up'] + 2 * jcfg['ir_up']
				)
			)
		self.data_bin = {
			'delay' 	: delay,
			'photon' 	: photon,
			'primary_energy' 	: primary_energy,
			'auger_energy'		:	auger_energy,
			'primary_angle'		: primary_angle,
			'auger_angle'			: auger_angle,
			'up'							: up
		}
		for hist_name in self.data_bin:
			hist = np.histogram(
				[],
		 		self.data_bin[hist_name][0],
				self.data_bin[hist_name][1],
				)
			self.data_hist[hist_name] = {
				'bins' : hist[1][:-1],
				'counts' : hist[0]
				}

		return

	def load_config(self,configfile):
		"""
		---------------------------------------------------------------------------
		---------------------------------------------------------------------------
			CONFIGURATION LOADER
		"""
		self.config = yaml.load(file(configfile))
			# instantiate ananlyzer
		if( 'analyzer' in self.config):
			# user specified analyzer
			ana_module = __import__(self.config['analyzer'])
		else:
			# default analyzer (does nothing)
			ana_module = __import__("ana_base")
		
		self.analyzer = ana_module.analyzer()
		return
		

	def build_shot_params(self):
		"""
		---------------------------------------------------------------------------
		---------------------------------------------------------------------------
		GENERATE SHOT PARAMETERS
		"""
		
		print('Setting up shot parameters')
		#set up the shot list
		c = self.config
		
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
		up_jitter_array =np.random.normal(
			1.0,
			c['jitter']['ir_up'],
			event_count
			)

		#combine all dsitributions in master shot parameter array
		self.shot_params={
			'time_delay': scan_array,
			'time_jitter' : timing_jitter_array,
			'photon_jitter' : photon_jitter_array,
			'up_jitter' : up_jitter_array,
			}
		return

	def build_hit_params(self):
		"""
		---------------------------------------------------------------------------
		---------------------------------------------------------------------------
		DEFINE HIT PARAM GENERATORS
		"""
		# Primary ionization time (xfel pulse width)
		print('setting up hit randomization callbacks')
		c = self.config
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

		def auger_angle(n):
			return np.random.rayleigh(
				c['atom']['auger_full_angle']*(np.pi/180),
				n
			)

		def primary_angle(n):
			return np.random.rayleigh(
				c['atom']['primary_full_angle']*(np.pi/180),
				n
			)

		def laser_up(n):
			return c['ir_pulse']['up']-np.random.rayleigh(
					c['ir_pulse']['up_variance'],
					n
				)

		#lambda expressions to extract hit randomization
		self.hit_params = {
			'primary_time_offset' 	: primary_time_offest,
			'primary_energy_offset' : primary_energy_offset,
			'primary_angle'					: primary_angle,
			'auger_time_offest' 		: auger_time_offest,
			'auger_energy' 					: auger_energy,
			'auger_angle'						: auger_angle,
			'ir_up'									: laser_up
		}

		return


	
	def streak(self,E_i,Up,omeg,Th,t):
		"""
		---------------------------------------------------------------------------
		---------------------------------------------------------------------------
		classical streaking model	
		"""
		return E_i + 2 * Up * ( np.sin( omeg * t )**2 ) * np.cos( 2 * Th ) +\
					np.sqrt( 8 * E_i * Up ) * np.sin( omeg * t ) * np.cos( Th )

	#function to model time evolution of IR Laser Up	
	def ir_up_env(self,t,d,up):
		"""
		---------------------------------------------------------------------------
		---------------------------------------------------------------------------
		Laser pulse envelope
		"""
		cw = ( d )/( 2 * np.sqrt( 2 * np.log(2) ) ) #implement FHWM
		return up * ( np.exp( -1*( (t)**2 ) / (2*(cw**2)) ) )**2 #gauss

	def sim(self):
		"""
		---------------------------------------------------------------------------
		---------------------------------------------------------------------------
		Main Simulation Routine
		"""
		speed_c = codata.physical_constants[ 'speed of light in vacuum'][0]
		#this is needed in rad/fs
		omeg = 2 * np.pi * speed_c / 		\
			( self.config['ir_pulse']['wavelength'] * 1e-6 ) * (1e-15)


		#prepare the analyzer
		self.analyzer.beforeRun( self.config )

		print("starting run")
		print("\n\n -------------------------------")
		for i in range( 0 , len(self.shot_params['time_delay']) ):
			# for each time step, create some ionization events

			# the following params come from a pre-randomized array
			# the reflect vars that stay fixed within one shot

			# current ir/fel time offset
			time = self.shot_params['time_delay'][i] + self.shot_params['time_jitter'][i]
			#run_delays = np.append( run_delays, time )

			# current ir amplitude (scale of 1.0)
			up = self.shot_params['up_jitter'][i]
			#print(up)
			
			# the central photon enegry of the xfel pulse
			photon = self.shot_params['photon_jitter'][i]
			#run_photons = np.append( run_photons, photon )

			# make the params that fluctuate with each hit
			# run the distribution functions passed in as "hits"
			# the argument is the number of values to generate
			# these statements create and operate on arrays

			p_time = time + self.hit_params['primary_time_offset']( self.config['atom']['primary_hits'] )
	 		

	 		p_angle = self.hit_params['primary_angle']( self.config['atom']['primary_hits'] )

 		
		
			p_energy = \
					self.hit_params['primary_energy_offset']( self.config['atom']['primary_hits'] ) - \
					self.config['atom']['primary_binding_energy']+ \
					photon
		
			# max up based on spatial profile of laser beam
			p_up_max = self.hit_params['ir_up'](self.config['atom']['primary_hits'] )


			# time modified up, based on laser envelope timing
			p_up = p_up_max * up * self.ir_up_env( 
					p_time, \
					self.config['ir_pulse']['duration'], \
					1.0\
				)


			primary_electrons = self.streak( p_energy, p_up,omeg, p_angle, time )
			# primary_elecrons is the array of 
			# electron energies "detected" in this shot

			# time is ir/fel delay + random val from decay distr.
			a_time = \
				time + \
				self.hit_params['primary_time_offset']( self.config['atom']['auger_hits'] ) + \
				self.hit_params['auger_time_offest']( self.config['atom']['auger_hits'] )

			a_angle = self.hit_params['auger_angle']( self.config['atom']['auger_hits'] )
			# energy is rand val within line width
			a_energy = self.hit_params['auger_energy']( self.config['atom']['auger_hits'] )

			# max up based on spatial profile of laser beam
			a_up = self.hit_params['ir_up'](self.config['atom']['auger_hits'] )

			# time modified up, based on laser envelope timing
			a_up = a_up * up * self.ir_up_env( 
					a_time, 
					self.config['ir_pulse']['duration'], 
					1.0\
				)

			#final energy of each auger electron

			auger_electrons = self.streak( a_energy, a_up, omeg, a_angle, a_time )

			# these are shot fixed enviroment varibales for the analyzer
			event_env={
				'photon' : photon,# "measured" photon energy
				'time_delay' : self.shot_params['time_delay'][i], #set time delay
				'time_jitter' : self.shot_params['time_jitter'][i] #"measured" jitter position
			}
			# send event electrons and env variables to analyzer
			self.analyzer.event(primary_electrons,auger_electrons,event_env)

			#accumulate in histograms
			self.fill_hist('auger_energy'		,auger_electrons 	)
			self.fill_hist('primary_energy'	,primary_electrons)
			self.fill_hist('delay'					,time 						)
			self.fill_hist('photon'					,photon 					)
			self.fill_hist('primary_angle'	,p_angle	 				)
			self.fill_hist('auger_angle'		,a_angle					)
			self.fill_hist('up' 						,p_up_max					)


			if( (i%100)==0 ):
				self.print_status(100.0*float(i)/float(len(self.shot_params['time_delay'])))

			if( (i%1000)==0 ):
				self.plot_prog( )
				self.analyzer.check()
				
			
			#END for each shot
		#signal run completion to analyzer
		self.analyzer.afterRun()
		self.wait(' run complete!')
		
		return

	def print_status(self,percent):
		"""
		---------------------------------------------------------------------------
		---------------------------------------------------------------------------
		status update
		"""
		status = r"Running: %d Percent | " % (percent)
		sys.stdout.write('\r' + status)
		sys.stdout.flush() # important

	# wait function
	def wait(self, str=None, prompt='Press return to continue...\n'):
		"""
		---------------------------------------------------------------------------
		---------------------------------------------------------------------------
		wait for return
		"""
		if str is not None:
			print str
		return raw_input(prompt)
	
	def fill_hist(self,hist_name,values):
		"""
		---------------------------------------------------------------------------
		---------------------------------------------------------------------------
		fill values into the desired histogram
		"""
		if hist_name not in self.data_hist:
			return

		bins = self.data_bin[hist_name]

		hist = np.histogram(
				values,
		 		bins[0],
				bins[1],
				)
		counts = self.data_hist[hist_name]['counts'] + hist[0]
		bins = hist[1][:-1]
		self.data_hist[hist_name]={
			'bins' : bins,
			'counts' : counts
		}


		return

	def plot_prog( self ):
		"""
		---------------------------------------------------------------------------
		---------------------------------------------------------------------------
		PLOT ACCUMULATED DATA
		"""
		
		fig = plt.figure(1)
		fig.subplots_adjust(hspace=.5)
		plt.clf()
		plt.title('Simulation progress')

		plt.subplot2grid((3,2),(0,0))
		plt.plot( 
			self.data_hist['primary_energy']['bins'],
			self.data_hist['primary_energy']['counts'] ,
			lw = 2 )
		plt.title("Primary Spectrum (Cumulative)")
		plt.xlabel('Photoelectron Energy [ eV ]')
		plt.ylabel('Counts')

		plt.subplot2grid((3,2),(0,1))
		plt.plot( 
			self.data_hist['auger_energy']['bins'],
			self.data_hist['auger_energy']['counts'] ,
			lw = 2 )
		plt.title("Auger Spectrum (Cumulative)")
		plt.xlabel('Photoelectron Energy [ eV ]')
		plt.ylabel('Counts')

		plt.subplot2grid((3,2),(1,0))
		plt.plot( 
			self.data_hist['delay']['bins'],
			self.data_hist['delay']['counts'] ,
			)
		plt.title("IR/FEL Delay")
		plt.xlabel('Delay [ fs ]')
		plt.ylabel('Counts')

		plt.subplot2grid((3,2),(1,1))
		plt.plot(
			self.data_hist['photon']['bins'],
			self.data_hist['photon']['counts'] ,
			)
		plt.title( "XFEL Central Photon Energy" )
		plt.xlabel(' Photon Energy [ eV ]')
		plt.ylabel('Counts')

		plt.subplot2grid((3,2),(2,0))

		plt.plot( 			
			self.data_hist['primary_angle']['bins']*(180/np.pi),
			self.data_hist['primary_angle']['counts'],
			lw=2)
		plt.plot(
			self.data_hist['auger_angle']['bins']*(180/np.pi),
			self.data_hist['auger_angle']['counts'],
		 	lw=2)
		plt.legend( ['Primary','Auger'] )
		plt.title( "Emission Angles")
		plt.xlim(0,90)
		plt.xlabel(' Angle [deg]')
		

		plt.subplot2grid((3,2),(2,1))
		plt.plot( 			
			self.data_hist['up']['bins'],
			self.data_hist['up']['counts'],\
			 lw=2)
		plt.title( "IR Intenisty")
		plt.xlabel('Peak Up [eV]')

		plt.draw()
		return

################################################################################
##
## END: py-auger engine class
################################################################################
################################################################################



##-----------------------------------------------------------------------------
##-----------------------------------------------------------------------------
##this happens on load
if __name__ == "__main__":
	parser = OptionParser()

	parser.add_option("-s", "--settings", dest="settings",
	                  help="use these settings for the simulation", metavar="SETTINGS")

	(options, args) = parser.parse_args()
	if(options.settings):
		configfile = options.settings
	else:
		configfile = 'default.yaml'
	
	sim_engine = py_auger_engine()
	sim_engine.setup( configfile )
	sim_engine.sim()










