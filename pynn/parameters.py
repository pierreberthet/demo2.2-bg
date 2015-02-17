# #############
#  PARAMETERS
# #############


inactive_state_rate = 400.   # stimulus rate for inactive cortex neurons, in Hz
active_state_rate = 1200.     # stimulus rate for active cortex neurons, in Hz

n_states = 3    #number of  states
m_actions = 3   #number of  actions

initial_state = 0

n_trials = 10   #number of trials per block
n_blocks = 1   #number of blocks



n_cortex_cells = 50
n_msns = 30
n_gpi = 10

cortex_input_weight = .1e-3          # uS
#cortex_input_weight = 5.            # uS
cortex_input_delay = 1.             # ms


#initial_weight_value =  1.e-03      # uS    initial weight value of the cortico-stratal connections, D1 and D2

wd1 = 1.e-03    # uS
wd2 = 1.e-03    # uS
std_wd1 = 1.e-04
std_wd2 = 1.e-04


gpi_external_rate = 800.       # external input rate for GPI, in Hz
gpi_external_weight = .4e-3    # external weight for GPI, in uS
gpi_external_delay = 1.         # external delay for GPI, in mS

d1_gpi_weight = .1e-3
d2_gpi_weight = .1e-3
d1_gpi_delay = 1.
d2_gpi_delay = 1.
# connection probabilities between individual neurons striatum - gpi
d1_gpi_prob = 1.0
d2_gpi_prob = 1.0

d1_lat_inh_prob = 0.3
d2_lat_inh_prob = 0.3
d1_lat_inh_weight = .2e-3
d2_lat_inh_weight = .2e-3
d1_lat_inh_delay = 1.
d2_lat_inh_delay = 1.

ctx_strd1_delay = 1.
ctx_strd2_delay = 1.
std_ctx_strd1_delay = .1
std_ctx_strd2_delay = .1




change = .1e-03 #fixed value for the weight update



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


sim_duration = 100
#sim_duration = 1000
time_step = 0.1

conn_filename = "full_conn_list.dat"
spike_gpi_fn = "out/gpi_spike_count.dat"
reward_fn = "out/reward.txt"
