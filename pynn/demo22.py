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

delay_ctx_strd1 = 1.
delay_ctx_strd2 = 1.

change = .3   #fixed value for the weight update


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


for block in xrange(n_blocks):
    for trial in xrange(n_trials):
        pass

	#sim
    #update_weights

##LOOP
#run 1 trial
#	set one poisson generator to active firing rate for one state in cortex
#	sim.run(time)
#	get spikes from GPi/SNr
#	offline computations of selection, reward, and update
#	load weights
#	new trial


def offline_update(i_state,  spike_trains):
    
    #Update the weights based on the state and the output spike trains
    #TODO  convert C code to python
    #TODO  select action based on output spike trains 
    #TODO  reward function that computes the reward given the curernt state and the selected action
    #TODO  store weights value for all the trials
    #TODO  decide how to set kappa and how to artificially increase the traces of the selected action
    #TODO  update function
    #TODO  initialize weights matrices, D1 and D2
    
  #  #local
  #  ##########
  #  epsilon =.0001
  #  #spike_height
  #  #initial value for the traces
  #  resolution = 1.
  #  fmax = 30.
  #  taui_ = 5.
  #  tauj_ = 6.
  #  taue_ = 50.
  #  taup_ = 200.
  #  
  #  w_d1 = np.array()

  #  # Primary synaptic traces
  #  zi_ += (spike_height * yi_ - zi_ + epsilon_ ) * resolution / taui_
  #  zj_ += (K_ < 0) ? (1./1000.0 - yj_/fmax_ - zj_ + epsilon_) * resolution / tauj_  : (spike_height * yj_ - zj_ + epsilon_ ) * resolution / tauj_

  #  # Secondary synaptic traces
  #  ei_  += (zi_ - ei_) * resolution / taue_
  #  ej_  += (zj_ - ej_) * resolution / taue_
  #  eij_ += (zi_ * zj_ - eij_) * resolution / taue_

  #  # Tertiary synaptic traces. Commented is from Wahlgren paper. */
  #  pi_  +=std::abs( K_) * (ei_ - pi_) * resolution / taup_/* * eij_*/
  #  pj_  +=std::abs( K_) * (ej_ - pj_) * resolution / taup_/* * eij_*/
  #  pij_ +=std::abs( K_ )* (eij_ - pij_) * resolution / taup_/* * eij_*/


    #conn_list = (pre_idx, post_idx, weight, delay)
    if got_reward(i_state, i_action, block):
        #increase D1 weights of selected action
	update_weights(i_state, i_action, 'D!')
    else:
	#increase D2 weights of selected action
	update_weights(i_state, i_action, 'D2')


    return


def update_weights(state, action, pathway):
    #increase weights between current state and selected action, decrease weights between inactive state and selected action
    #and between current state and non selected action in the specified pathway. Other pathway is left untouched.
    for s in xrange(n_states):
        w = np.loadtxt(pathway+"_state"+str(s)+"_to_action"+str(action)+".dat")
        if s==state:
	    #increase weights
	    w += change
        else:
            #decrease weights
	    w-+ change/(n_states-1.)
	w = w/sum(w)   #normalize values

    for a in xrange(n_actions):
        if a!=action:
            w = np.loadtxt(pathway+"_state"+str(state)+"_to_action"+str(a)+".dat")
            #decrease weights
	    w-+ change/(n_states-1.)
	    w = w/sum(w)   #normalize values

    return


def init_weights(n_states, n_actions):
    for pathway in {"D1", "D2"}:    
        if pathway=="D1":
            delay = delay_ctx_strd1
        else:
            delay = delay_ctx_strd2
        for s in xrange(n_states):
            for a in xrange(n_actions):
                w = np.ones((n_cortex_cells, n_msns))*1.  	 #to change: implement variability
                conn_list = []
                for pre in xrange(n_cortex_cells):
		    for post in xrange(n_msns):
			conn_list.append(pre, post, w[pre, post], delay)  
	        np.savetxt(pathway+"_state"+str(s)+"_to_action"+str(a)+".dat", conn_list)
    return

def make_single_file():
    

   np.savetxt()
   return


def got_reward(state, action, block):
    rew = ( action == ((block + state) % n_actions) )
    return rew


