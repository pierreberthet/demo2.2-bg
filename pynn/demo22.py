import pyNN.nest as sim
import numpy as np
from parameters import *
from matplotlib import pyplot as plt

# #####################################################
#     
#    OFFLINE COMPUTATIONS OF BG UPDATES
#
# ####################################################


def offline_update(i_state, i_action, block):
    
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

    #output_rates = np.loadtxt(spike_count_full_filename)
    #output_rates = np.loadtxt(spike_gpi_fn)
    #i_action = np.argmax(output_rates)

    rew = 0
    #conn_list = (pre_idx, post_idx, weight, delay)
    if got_reward(i_state, i_action, block):
        #increase D1 weights of selected action
        rew = 1
        #print 'reward ', rew, 'D1 '
       	update_weights(i_state, i_action, 'D1')
    else:
       	#increase D2 weights of selected action
        #print 'reward ', rew, 'D2 ' 
       	update_weights(i_state, i_action, 'D2')
    return rew


def update_weights(state, action, pathway):
    #increase weights between current state and selected action, decrease weights between inactive state and selected action
    #and between current state and non selected action in the specified pathway. Other pathway is left untouched.
    for s in xrange(n_states):
        temp = np.loadtxt(pathway+"_state"+str(s)+"_to_action"+str(action)+".dat")
        if s==state:
	        #increase weights
            #temp[:,2] += change
            temp[:,2] += (w_uplimit - temp[:,2]) * learning_rate
            #print 'PLUS', pathway+"_state"+str(s)+"_to_action"+str(action)
        else:
            #decrease weights
            #temp[:,2]-= change/(n_states-1.)
            temp[:,2] -= (temp[:,2]-w_lowlimit) * learning_rate
            #print 'MINUS', pathway+"_state"+str(s)+"_to_action"+str(action)
        #temp[:,2] = temp[:,2]/sum(temp[2,:])   #normalize values
        np.savetxt(pathway+"_state"+str(s)+"_to_action"+str(action)+".dat",temp )

    for a in xrange(m_actions):
        if a!=action:
            temp = np.loadtxt(pathway+"_state"+str(state)+"_to_action"+str(a)+".dat")
            #decrease weights
            temp[:,2] -= change/(n_states-1.)
            #temp[:,2] = temp[:,2]/sum(temp[2,:])   #normalize values
            np.savetxt(pathway+"_state"+str(state)+"_to_action"+str(a)+".dat",temp )

    make_single_file()

    return


def init_weights():
    #initialize the weights of the cortico-striatal connections
    #TODO add weight variability
    #TODO normalize weights?
    for s in xrange(n_states):
        #print 'S', s
        for a in xrange(m_actions):
            #print 'A', a
            conn_list_D1 = []
            conn_list_D2 = []
            for pre in xrange(n_cortex_cells*s, n_cortex_cells*s+n_cortex_cells):
                #print 'pre  ', pre
                for post in xrange(n_msns*a, n_msns*a+n_msns):  #gids_d1[a]:
                    #print 'post ', post
                    conn_list_D1.append((pre, post, np.round(np.random.normal(wd1, std_wd1),4), np.round(np.random.normal(ctx_strd1_delay, std_ctx_strd1_delay), 1)))  
                    #conn_list_D1.append((pre, post, np.round(np.random.normal(wd1, std_wd1),4), np.round(np.random.normal(ctx_strd1_delay, std_ctx_strd1_delay), 1)))  
                for post in xrange((a+m_actions)*n_msns, (a+m_actions)*n_msns+n_msns): #gids_d2[a]:
                    #print 'post ', post
                    conn_list_D2.append((pre, post, np.round(np.random.normal(wd2, std_wd2),4), np.round(np.random.normal(ctx_strd2_delay, std_ctx_strd2_delay), 1)))  
                    #conn_list_D2.append((pre, post, np.round(np.random.normal(wd2, std_wd2),4), np.round(np.random.normal(ctx_strd2_delay, std_ctx_strd2_delay), 1)))  
            #for pre in gids_cortex[s]:
            #    for post in gids_d1[a]:
            #        conn_list_D1.append((pre, post, np.round(np.random.normal(wd1, std_wd1),4), np.round(np.random.normal(ctx_strd1_delay, std_ctx_strd1_delay), 1)))  
            #    for post in gids_d2[a]:
            #        conn_list_D2.append((pre, post, np.round(np.random.normal(wd2, std_wd2),4), np.round(np.random.normal(ctx_strd2_delay, std_ctx_strd2_delay), 1)))  
            np.savetxt("D1_state"+str(s)+"_to_action"+str(a)+".dat", conn_list_D1)
            np.savetxt("D2_state"+str(s)+"_to_action"+str(a)+".dat", conn_list_D2)
            
           # # Old connections file, without variability in the weights and delays  
           # for pre in gids_cortex[s]:
	       #     for post in gids_d1[a]:
	       #         conn_list.append((pre, post, w[gids_cortex[s].index(pre), gids_d1[a].index(post)], ctx_strd1_delay))  
	       #         np.savetxt("D1_state"+str(s)+"_to_action"+str(a)+".dat", conn_list)
	       #     for post in gids_d2[a]:
	       #         conn_list.append((pre, post, w[gids_cortex[s].index(pre), gids_d2[a].index(post)], ctx_strd2_delay))  
	       #         np.savetxt("D2_state"+str(s)+"_to_action"+str(a)+".dat", conn_list)

    make_single_file()

    print "INIT OK"
    return


def make_single_file():
    #merge all the connections files into one
    #order in the lsit can be edited by changing order of the for loops below
    single = np.array([[0,0,0,0]])
    for s in xrange(n_states):
        for pathway in {"D1", "D2"}:    
            for a in xrange(m_actions):
                tempp =np.loadtxt(pathway+"_state"+str(s)+"_to_action"+str(a)+".dat")
                single = np.concatenate((single, tempp), axis=0)
                #plt.scatter(tempp[:,0], tempp[:,1], label="pre "+pathway+str(s)+str(a))
                #plt.scatter(tempp[:,1],  label="post "+pathway+str(s)+str(a))
    
    #plt.plot(single[:,0], single[:,1])
    single = single[1:,:]
    np.savetxt(conn_filename, single)
    
    # columns = ["i", "j", "weight", "delay"]
    #plt.plot(np.loadtxt("full_conn_list.dat")[:,2])
    #plt.legend()
    #plt.show()

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
