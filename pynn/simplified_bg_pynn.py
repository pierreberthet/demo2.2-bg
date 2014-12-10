import pyNN.nest as sim
#import pyNN.neuron as sim
import numpy as np
import sys

from parameters import *


def save_spikes(pop_list, base_name, filename):
    for idx, pop in enumerate(pop_list):
        spikes = pop.getSpikes()
        fname = base_name + 'spikes_' + str(idx) + '_' + filename
        print "saving spikes to file:", fname, "length=", len(spikes)
        np.savetxt(fname, np.array(spikes, ndmin=2))

if __name__ == '__main__':
    output_base = "out/"
    spike_count_filename = "gpi_spike_count.dat"

    weight_filename = conn_filename    # filename, from which the cortex - striatum connections are read

    spike_count_full_filename = output_base + spike_count_filename

    active_state = int(sys.argv[1])

    #Model of the basal ganglia D1 and D1 pathways. States and actions are populations coded.

    import pyNN
    pyNN.utility.init_logging(None, debug=True)

    sim.setup(time_step)

    # cell class for all neurons in the network
    # (on HMF can be one of IF_cond_exp, EIF_cond_exp_isfa_ista)
    cellclass = sim.IF_cond_exp


    # #############
    #  POPULATIONS
    # #############
    #CORTEX input population: N states, poisson inputs

    #?assemblies of m_actions populations or dictionnary of populations?
    #STRIATUM 2 populations of M actions, D1 and D2

    #GPi/SNr 1 population of M actions, baseline firing rate driven by external poisson inputs


    cortex = [
        sim.Population(n_cortex_cells, cellclass, neuron_parameters, label="CORTEX_{}".format(i))
        for i in xrange(n_states)]

    cortex_assembly = sim.Assembly(
        *cortex,
        label="CORTEX")

    # independent Poisson input to cortex populations.
    # /active_state/ determines, which population receives
    # a different firing rate
    cortex_input = []
    for i in xrange(n_states):

        if i == active_state:
            rate = active_state_rate
        else:
            rate = inactive_state_rate

        new_input = sim.Population(
            n_cortex_cells,
            sim.SpikeSourcePoisson,
            {'rate': rate},
            label="STATE_INPUT_" + str(i))
        sim.Projection(
            new_input,
            cortex[i],
            sim.OneToOneConnector(
                weights=cortex_input_weight))

        cortex_input.append(new_input)

    # striatum:
    # exciatatory populations
    striatum_d1 = [
        sim.Population(n_msns, cellclass, neuron_parameters, label="D1_{}".format(i))
        for i in xrange(m_actions)]

    # inhibitory populations
    striatum_d2 = [
        sim.Population(n_msns, cellclass, neuron_parameters, label="D2_{}".format(i))
        for i in xrange(m_actions)]

    # Striatum D2->D2 and D1->D1 lateral inhibition
    for lat_inh_source in xrange(m_actions):
        for lat_inh_target in xrange(m_actions):
            if lat_inh_source == lat_inh_target:
                continue
            sim.Projection(
                striatum_d1[lat_inh_source],
                striatum_d1[lat_inh_target],
                sim.FixedProbabilityConnector(
                    d1_lat_inh_prob,
                    weights=d1_lat_inh_weight,
                    delays=d1_lat_inh_delay),
                target="inhibitory",
                label="d1_lateral_inhibition_{}_{}".format(
                    lat_inh_source, lat_inh_target))
            sim.Projection(
                striatum_d2[lat_inh_source],
                striatum_d2[lat_inh_target],
                sim.FixedProbabilityConnector(
                    d2_lat_inh_prob,
                    weights=d2_lat_inh_weight,
                    delays=d2_lat_inh_delay),
                target="inhibitory",
                label="d2_lateral_inhibition_{}_{}".format(
                    lat_inh_source, lat_inh_target))

    striatum_assembly = sim.Assembly(
        *(striatum_d1 + striatum_d2),
        label="STRIATUM")

    # cortex - striatum connection, all-to-all using loaded weights
    sim.Projection(
        cortex_assembly,
        striatum_assembly,
        sim.FromFileConnector(
            weight_filename))

    gpi = [
        sim.Population(n_gpi, cellclass, neuron_parameters,
                       label="GPI_{}".format(i))
        for i in xrange(m_actions)
        ]
    gpi_assembly = sim.Assembly(
        *gpi,
        label="GPi")

    # external Poisson input to GPi
    gpi_input = sim.Population(
        m_actions * n_gpi,
        sim.SpikeSourcePoisson,
        dict(
            duration=sim_duration,
            rate=gpi_external_rate,
            start=0.),
        label="GPI_EXT_INPUT")
    sim.Projection(
        gpi_input,
        gpi_assembly,
        sim.OneToOneConnector(
            weights=gpi_external_weight))

    # striatum - gpi connections
    for i in xrange(m_actions):
        sim.Projection(
            striatum_d1[i],
            gpi[i],
            sim.FixedProbabilityConnector(d1_gpi_prob, weights=d1_gpi_weight))

        sim.Projection(
            striatum_d2[i],
            gpi[i],
            sim.FixedProbabilityConnector(d2_gpi_prob, weights=d2_gpi_weight),
            target="inhibitory")

    cortex_assembly.record()
    striatum_assembly.record()
    gpi_assembly.record()

    sim.run(sim_duration)

    save_spikes(cortex, output_base, "cortex.dat")
    save_spikes(striatum_d1, output_base, "striatum_d1.dat")
    save_spikes(striatum_d2, output_base, "striatum_d2.dat")
    save_spikes(gpi, output_base, "striatum.dat")

    output_rates = np.array(
        [len(i.getSpikes()) for i in gpi])
    np.savetxt(spike_count_full_filename, output_rates)
    sim.end()


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


