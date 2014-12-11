import pyNN.nest as sim
import numpy as np
from parameters import *

# #####################################################
#     
#    OFFLINE COMPUTATIONS OF BG UPDATES
#
# ####################################################






def offline_update(i_state, block,  spike_count_full_filename):
    
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


    # ###################
    #  ACTION SELECTION
    # ###################
    #INITAL TEST
    #TODO softmax implementation

    output_rates = np.loadtxt(spike_count_full_filename)
    i_action = np.argmax(output_rates)

    #conn_list = (pre_idx, post_idx, weight, delay)
    if got_reward(i_state, i_action, block):
        #increase D1 weights of selected action
       	update_weights(i_state, i_action, 'D1')
    else:
       	#increase D2 weights of selected action
       	update_weights(i_state, i_action, 'D2')

    return


def update_weights(state, action, pathway):
    #increase weights between current state and selected action, decrease weights between inactive state and selected action
    #and between current state and non selected action in the specified pathway. Other pathway is left untouched.
    for s in xrange(n_states):
        temp = np.loadtxt(pathway+"_state"+str(s)+"_to_action"+str(action)+".dat")
        if s==state:
	        #increase weights
            temp[:,2] += change
        else:
            #decrease weights
            temp[:,2]-= change/(n_states-1.)
        #temp[:,2] = temp[:,2]/sum(temp[2,:])   #normalize values
        np.savext(pathway+"_state"+str(s)+"_to_action"+str(action)+".dat",temp )

    for a in xrange(m_actions):
        if a!=action:
            temp = np.loadtxt(pathway+"_state"+str(state)+"_to_action"+str(a)+".dat")
            #decrease weights
            temp[:,2] -= change/(n_states-1.)
            #temp[:,2] = temp[:,2]/sum(temp[2,:])   #normalize values
            np.savext(pathway+"_state"+str(s)+"_to_action"+str(action)+".dat",temp )

    return


def init_weights(n_states, m_actions):
    #initialize the weights of the cortico-striatal connections
    #TODO add weight variability
    #TODO normalize weights?
    for pathway in {"D1", "D2"}:    
        if pathway=="D1":
            delay = ctx_strd1_delay 
        else:
            delay = ctx_strd2_delay
        for s in xrange(n_states):
            for a in xrange(m_actions):
                w = np.ones((n_cortex_cells, n_msns))*1.  	 #to change: implement variability
                conn_list = []
                for pre in xrange(n_cortex_cells):
		            for post in xrange(n_msns):
		                conn_list.append((pre, post, w[pre, post], delay))  
	                    np.savetxt(pathway+"_state"+str(s)+"_to_action"+str(a)+".dat", conn_list)

    make_single_file()

    return

def make_single_file():
    #merge all the connections files into one
    #order in the lsit can be edited by changing order of the for loops below
    single = np.array([[0,0,0,0]])
    for s in xrange(n_states):
        for pathway in {"D1", "D2"}:    
            for a in xrange(m_actions):
	            single = np.concatenate((single, np.loadtxt(pathway+"_state"+str(s)+"_to_action"+str(a)+".dat")), axis=0)
    single = single[1:,:]
    np.savetxt(conn_filename, single)
    return


def got_reward(state, action, block):
    #compute the reward based on the current state and the selected action, and the learning block
    rew = ( action == ((block + state) % m_actions) )
    return rew






# ###########################
#
#        DESCRIPTION
#
#############################




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

#def set_state(i_state):
#    #set the poisson generator of the specified state coding population in cortex to high activity
#    #set the other state coding populations to low activity
#
#    #dummy function
#    #TODO write the correct function
#    for s in xrange(n_states):
#        sim.set_poisson(s,inactive_state_rate )
#    sim.set_poisson(i_state, active_state_rate)
#
#    return







