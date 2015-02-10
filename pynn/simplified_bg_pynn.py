import pyNN.nest as sim
#import pyNN.neuron as sim
import pyNN.utility
import numpy as np
import sys

from parameters import *
from matplotlib import pyplot as plt

def save_spikes(pop_list, base_name, filename):
    for idx, pop in enumerate(pop_list):
        print idx, pop
        spikes = pop.getSpikes()
        print 'spikes', spikes.list_units
        print "////", spikes.list_recordingchannels
        fname = base_name + 'spikes_' + str(idx) + '_' + filename
        print "saving spikes to file:", fname, "length=", len(spikes)
        np.savetxt(fname, np.array(spikes, ndmin=2))

##if __name__ == '__main__':
def run(a_state):
    output_base = "out/"
    spike_count_filename = "gpi_spike_count.dat"

    weight_filename = conn_filename    # filename, from which the cortex - striatum connections are read

    spike_count_full_filename = output_base + spike_count_filename

    #active_state = int(sys.argv[1])
    active_state = a_state

    #Model of the basal ganglia D1 and D1 pathways. States and actions are populations coded.

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
            sim.OneToOneConnector(),
            sim.StaticSynapse(weight=cortex_input_weight, delay=cortex_input_delay)
            )

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
                    d1_lat_inh_prob),
                    sim.StaticSynapse(
                        weight=d1_lat_inh_weight,
                        delay=d1_lat_inh_delay),
                receptor_type="inhibitory",
                label="d1_lateral_inhibition_{}_{}".format(
                    lat_inh_source, lat_inh_target))
            sim.Projection(
                striatum_d2[lat_inh_source],
                striatum_d2[lat_inh_target],
                sim.FixedProbabilityConnector(
                    d2_lat_inh_prob),
                    sim.StaticSynapse(
                        weight=d2_lat_inh_weight,
                        delay=d2_lat_inh_delay),
                receptor_type="inhibitory",
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
            sim.StaticSynapse(
                weight=gpi_external_weight,
                delay= gpi_external_delay)))

    # striatum - gpi connections
    for i in xrange(m_actions):
        sim.Projection(
            striatum_d1[i],
            gpi[i],
            sim.FixedProbabilityConnector(d1_gpi_prob), 
            sim.StaticSynapse( weight=d1_gpi_weight, delay = d1_gpi_delay))

        sim.Projection(
            striatum_d2[i],
            gpi[i],
            sim.FixedProbabilityConnector(d2_gpi_prob),
            sim.StaticSynapse(weight=d2_gpi_weight, delay=d2_gpi_delay),
            #target="inhibitory")
            receptor_type="inhibitory")

    cortex_assembly.record('spikes')
    striatum_assembly.record('spikes')
    gpi_assembly.record('spikes')


    sim.run(sim_duration)
    sim.end()
    
    label = "CORTEX_0" 
    #print 'cortex get pop', cortex_assembly.get_population(label)
    #print 'cortex describe', cortex_assembly.describe()
    #cortex_assembly.write_data("spikes")
    #cortex_assembly.get_population(label).write_data("spikes")
    #spikes = gpi_assembly  #get_data("spikes", gather=True)
   # print "getdata spikes", spikes
   # print 'spikes.segment', spikes.segments
    #print 'spikes.segments.SpikeTrains', spikes.segments.spike

    #save_spikes(cortex_assembly, output_base, "cortex.dat")
    #save_spikes(striatum_d1, output_base, "striatum_d1.dat")
    #save_spikes(striatum_d2, output_base, "striatum_d2.dat")
    #save_spikes(gpi, output_base, "gpi.dat")

    #output_rates = np.array(
    #    [len(i.getSpikes()) for i in gpi])
    #np.savetxt(spike_count_full_filename, output_rates)
    
   # for seg in cortex_assembly.segments:
   #     print("Analyzing segment %d" % seg.index)
   #     stlist = [st - st.t_start for st in seg.spiketrains]
   #     plt.figure()
   #     count, bins = np.histogram(stlist)
   #     plt.bar(bins[:-1], count, width=bins[1] - bins[0])
   #     plt.title("PSTH in segment %d" % seg.index)
    cortex_mean_spikes = np.zeros(n_states)
    gpi_mean_spikes = np.zeros(m_actions)
    d1_mean_spikes = np.zeros(m_actions)
    d2_mean_spikes = np.zeros(m_actions)
    for i in xrange(n_states):
        cortex_mean_spikes[i] = cortex_assembly.get_population("CORTEX_"+str(i)).mean_spike_count()
    for i in xrange(m_actions):
        gpi_mean_spikes[i] = gpi_assembly.get_population("GPI_"+str(i)).mean_spike_count()
        d1_mean_spikes[i] = striatum_assembly.get_population("D1_"+str(i)).mean_spike_count()
        d2_mean_spikes[i] = striatum_assembly.get_population("D2_"+str(i)).mean_spike_count()

    print 'CORTEX ', cortex_mean_spikes
    print 'D1', d1_mean_spikes
    print 'D2', d2_mean_spikes

    return gpi_mean_spikes


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



