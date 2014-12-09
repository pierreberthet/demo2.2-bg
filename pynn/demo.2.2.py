import pyNN.nest as sim
import numpy as np

#Model of the basal ganglia D1 and D1 pathways. States and actions are populations coded.



# #############
#  PARAMETERS
# #############
n_states = 3	#number of  states
m_actions = 3   #number of  actions

n_cortex_cells = 50
n_msns 	= 30
n_gpi 	= 10


#neurons models and parameters

#params = []
neuron = sim.IF_cond_alpha(params)



# #############
#  POPULATIONS
# #############
#CORTEX input population: N states, poisson inputs

#?assemblies of n_actions populations or dictionnary of populations?
#STRIATUM 2 populations of M actions, D1 and D2


#GPi/SNr 1 population of M actions, baseline firing rate driven by external poisson inputs


# #############
# CONNECTIONS 
# #############

#N poisson generators to N states in cortex
#M poisson generators to M actions in GPi/SNr
#background noise for all meurons?


#all to all cortex to striatum for both D1 and D2 populations
#"plastic weights"

#lateral inhibition D2-D2 and D1-D1 
#static weights

#D1[m] --> GPi[m] positive static weight
#D2[m] --> GPi[m] negative static weight

# #############
# RECORDERS 
# #############

#spike detectors
#	GPi/SNr



# #############
# SIMULATION 
# #############

#initialize noise

##LOOP
#run 1 trial
#	set one poisson generator to active firing rate for one state in cortex
#	sim.run(time)
#	get spikes from GPi/SNr
#	offline computations of selection, reward, and update
#	load weights
#	new trial




