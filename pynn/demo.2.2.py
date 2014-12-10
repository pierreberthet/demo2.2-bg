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

Cm = 250.
Vreset = -75.
Vth = -50.
gL = 16.666

inactive_state_rate = 1000.
active_state_rate = 1600.




#neurons models and parameters

#params = []
neuron = sim.IF_cond_alpha(params)



# #############
#  POPULATIONS
# #############
#CORTEX input population: N states, poisson inputs

inputs = {}
for i in xrange(n_states):
    inputs[i] = sim.Population(n_cortex_cells, SpikeSourcePoisson, {'rate': inactive_state_rate}, "poisson_iput_"+str(i))


def set_state(i_state):
    for i in xrange(n_states):
	sim.SetStatus(inputs[i], {'rate':inactive_state_rate})
    sim.SetStatus(inputs[i_state], {'rate':active_State_rate})
    


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


def update_weights(i_state,  spike_trains):
    #Update the weights based on the state and the output spike trains
    #TODO  reward function that computes the reward given the curernt state and the selected action
    #TODO  store weights value for all the trials
    #TODO  decide how to set kappa and how to artificially increase the traces of the selected action
    #TODO  update function

    zi_ += (spike_height * yi_ - zi_ + epsilon_ ) * resolution / taui_;
    zj_ += (K_ < 0) ? (1./1000.0 - yj_/fmax_ - zj_ + epsilon_)
                * resolution / tauj_  : (spike_height * yj_ - zj_ + epsilon_ ) * resolution / tauj_;


    /* Secondary synaptic traces */
    ei_  += (zi_ - ei_) * resolution / taue_;
    ej_  += (zj_ - ej_) * resolution / taue_;
    eij_ += (zi_ * zj_ - eij_) * resolution / taue_;

    /* Tertiary synaptic traces. Commented is from Wahlgren paper. */
    pi_  +=std::abs( K_) * (ei_ - pi_) * resolution / taup_/* * eij_*/;
    pj_  +=std::abs( K_) * (ej_ - pj_) * resolution / taup_/* * eij_*/;
    pij_ +=std::abs( K_ )* (eij_ - pij_) * resolution / taup_/* * eij_*/;






