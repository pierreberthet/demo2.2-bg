# #############
#  PARAMETERS
# #############


inactive_state_rate = 1000.   # stimulus rate for inactive cortex neurons, in Hz
active_state_rate = 1600.     # stimulus rate for active cortex neurons, in Hz

n_states = 3    #number of  states
m_actions = 3   #number of  actions

n_cortex_cells = 50
n_msns = 30
n_gpi = 10

cortex_input_weight = 5e-3          # uS

gpi_external_rate = 1000.      # external input rate for GPI, in Hz
gpi_external_weight = 0.1e-3    # external weight for GPI, in uS

d1_gpi_weight = 1e-3
d2_gpi_weight = 1e-3
# connection probabilities between individual neurons striatum - gpi
d1_gpi_prob = 1.0
d2_gpi_prob = 1.0

d1_lat_inh_prob = 0.3
d2_lat_inh_prob = 0.3
d1_lat_inh_weight = 2e-3
d2_lat_inh_weight = 2e-3
d1_lat_inh_delay = 1.
d2_lat_inh_delay = 1.


# TODO: set remaining neuron parameters to non-default values:
# original parameters:
# Cm = 250.
# Vreset = -75.
# Vth = -50.
# gL = 16.666

neuron_parameters = {
    'cm': 0.25,                    # nF
    'e_rev_E': 0.0,
    'e_rev_I': -70.0,
    'i_offset': 0.0,
    'tau_m': 250. / 16.666,        # cm/gl, nF in ms
    'tau_refrac': 0.1,
    'tau_syn_E': 5.0,
    'tau_syn_I': 5.0,
    'v_reset': -75.0,
    'v_rest': -75.0,
    'v_thresh': -50.0
    }


sim_duration = 1000.
time_step = 0.1

