import pyNN.neuron as sim
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

    weight_filename = "weights.dat"    # filename, from which the cortex - striatum connections are read

    spike_count_full_filename = output_base + spike_count_filename

    active_state = int(sys.argv[1])

    #Model of the basal ganglia D1 and D1 pathways. States and actions are populations coded.

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
        sim.Population(n_cortex_cells, cellclass, neuron_parameters)
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
            label="poisson_iput_" + str(i))
        sim.Projection(
            new_input,
            cortex[i],
            sim.OneToOneConnector(
                weights=cortex_input_weight))

        cortex_input.append(new_input)

    # striatum:
    # exciatatory populations
    striatum_d1 = [
        sim.Population(n_msns, cellclass, neuron_parameters)
        for i in xrange(m_actions)]

    # inhibitory populations
    striatum_d2 = [
        sim.Population(n_msns, cellclass, neuron_parameters)
        for i in xrange(m_actions)]

    striatum_assembly = sim.Assembly(
        *(striatum_d1 + striatum_d2),
        label="STRIATUM")

    # TODO: cortex - striatum connection, all-to-all using loaded weights

    sim.Projection(
        cortex_assembly,
        striatum_assembly,
        sim.FromFileConnector(
            weight_filename))

    gpi = [
        sim.Population(n_gpi, cellclass, neuron_parameters)
        for i in xrange(m_actions)
        ]
    gpi_assembly = sim.Assembly(
        *gpi,
        label="GPI")

    # external Poisson input to GPI
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




