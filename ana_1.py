# 1s streak sorter
import numpy as np
import scipy as sp
# plotting
from matplotlib import pyplot as plt

from ana_base import *

class analyzer(analyzer):

	
	config = {}
	primary_data_hist = None
	auger_data_hist = None
	auger_time_hist = None
	primary_time_hist = None
	running = False
	def __init__(self):
		self.log('analyzer module loaded')
		return

	def log(self,s):
		print("ana1: "+s)
		return

	def beforeRun(self, config , dest_dir="."):
		self.log('preparing analyzer....')
		self.config = config
		#data configurations...
		dcfg = config['data']
		#set up the data histogram
		self.auger_data_bins = [ 
			dcfg['streak_bins'] ,
			dcfg['auger_bins'] 
			]
		self.auger_data_range = [
			( dcfg['streak_min'], dcfg['streak_max']),
			( dcfg['auger_min'] , dcfg['auger_max'] )
		]
		self.auger_data_hist=np.zeros(self.auger_data_bins)

		self.primary_data_bins = [
			dcfg['streak_bins'] ,
			dcfg['primary_bins'] 
		]
		self.primary_data_range = [
			( dcfg['streak_min'], dcfg['streak_max']		),
			( dcfg['primary_min'] , dcfg['primary_max'] )
		]
		self.primary_data_hist=np.zeros(self.primary_data_bins)
		self.running=True;
		return
		
	def event(self,primary_electrons,auger_electrons,env):
		#{'time_jitter': ##, 'photon': ##, 'time_delay': ##}
		if(len(auger_electrons)<1):
			return

		if(len(primary_electrons)<1):
			return

		avg_p = np.mean(primary_electrons)
		# streaking position
		streak = avg_p + \
			self.config['atom']['primary_binding_energy'] - \
			env['photon']

		# matched arrays of streaking positions 
		streak_a = streak * np.ones( len(auger_electrons	) )
		streak_p = streak * np.ones( len(primary_electrons) )
		

		self.fillPrimary( streak_p, primary_electrons )
		self.fillAuger(		streak_a, auger_electrons )

		return
	
	def fillAuger(self, values_x = [], values_y = [] ):
		#print('\n\n AUGER')
		#print values_x,values_y
		hist = np.histogram2d(
			values_x,
			values_y,
			bins = self.auger_data_bins,
			range = self.auger_data_range
			)
		#print(np.max(hist))
		#print('\n\n')
		self.auger_axis = hist[2]
		self.auger_data_hist = self.auger_data_hist + hist[0]
		#print('\n auger')
		#print(np.max(self.auger_data_hist))
		return

	def fillPrimary(self, values_x = [], values_y = [] ):
		#print('\n\n PRI')
		#print values_x,values_y
		
		hist = np.histogram2d(
			values_x,
			values_y,
			bins = self.primary_data_bins,
			range = self.primary_data_range
			)
		
		self.streak_axis = hist[1]
		self.primary_axis = hist[2]

		self.primary_data_hist =self.primary_data_hist+ hist[0]
		
		
		return


	def afterRun(self):
		return
	def stop(self):
		if(self.running):
			self.afterRun()
		return
	def check(self):
		self.plot_maps()
		self.plot_lineouts()


		return

	def plot_maps(self):
		#normailize the data by row
		norm_primary = np.transpose(self.primary_data_hist)
		norm_primary = norm_primary/np.sum(norm_primary,0)
		norm_primary = np.transpose(norm_primary)
		primary_max = np.mean(norm_primary)+3*np.std(norm_primary)

		norm_auger = np.transpose(self.auger_data_hist)
		norm_auger = norm_auger/np.sum(norm_auger,0)
		norm_auger = np.transpose(norm_auger)
		auger_max = np.mean(norm_auger)+3*np.std(norm_auger)
		

		fig = plt.figure(2)
		fig.subplots_adjust(hspace=.5)
		plt.clf()
		

		plt.subplot2grid((2,1),(0,0))
		plt.imshow(
			norm_primary,
			extent = (
				self.primary_data_range[1][0],
				self.primary_data_range[1][1],
				self.primary_data_range[0][0],
				self.primary_data_range[0][1],

				),
			aspect='auto'
			)
		plt.title('Sorted Primary')
		plt.xlabel('Photoelectron Energy [ eV ]')
		plt.ylabel('Detected Streak [ eV ] ')



		plt.subplot2grid((2,1),(1,0))
		plt.imshow(
			norm_auger,
			extent = (
				self.auger_data_range[1][0],
				self.auger_data_range[1][1],
				self.auger_data_range[0][0],
				self.auger_data_range[0][1],
				),
			clim = (0,auger_max),
			aspect='auto'
			)
		plt.title('Sorted Auger')
		plt.xlabel('Photoelectron Energy [ eV ]')
		plt.ylabel('Detected Streak [ eV ]')

		plt.draw()
		return

	def plot_lineouts(self):
		# lower half of auger data
		auger_lineout_count= np.floor(np.shape(self.auger_data_hist)[0]/2.0)
		# width of each lineout when making 4 lineouts
		auger_lineout_width = np.floor(auger_lineout_count/4.0)
		# 
		
		auger_lineouts =[
			np.sum( self.auger_data_hist[ -1*auger_lineout_width : -1 										],0 ),
			np.sum( self.auger_data_hist[ -2*auger_lineout_width : -1*auger_lineout_width ],0 ),
			np.sum( self.auger_data_hist[ -3*auger_lineout_width : -2*auger_lineout_width ],0 ),
			np.sum( self.auger_data_hist[ -4*auger_lineout_width : -0*auger_lineout_width ],0 ),
		]


		fig = plt.figure(3)
		
		plt.clf()

		
		plt.plot(self.auger_axis[:-1], auger_lineouts[3],lw=1)
		plt.plot(self.auger_axis[:-1], auger_lineouts[2],lw=1)
		plt.plot(self.auger_axis[:-1], auger_lineouts[1],lw=1)
		plt.plot(self.auger_axis[:-1], auger_lineouts[0],lw=2, c='black')
		plt.title('Sorted Auger Lineouts')
		plt.xlabel('Photoelectron Energy [ eV ]')
		plt.ylabel('Counts')

		plt.legend([
			str(self.streak_axis[-3*auger_lineout_width])+" eV",
			str(self.streak_axis[-2*auger_lineout_width])+" eV",
			str(self.streak_axis[-1*auger_lineout_width])+" eV",
			str(self.streak_axis[-1])+" eV",
			])
		plt.draw()

		return
