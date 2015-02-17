import demo22
from parameters import *
import simplified_bg_pynn as network
from matplotlib import pyplot as plt
import numpy as np



################
# INIT STORAGE #
################

reward = np.zeros(n_blocks*n_trials)







################
#    RUN       #
################


#demo22.init_weights(n_states, m_actions)
counter = 0
init= True

for block in xrange(n_blocks):
    for trial in xrange(n_trials):
        i_state = (trial+block) % n_states
        print 'TRIAL ', trial, 'BLOCK ', block
        gpi_spikes = network.run(i_state, init)
        init = False
        print 'GPI SPIKES ', gpi_spikes
        i_action = np.argmax(gpi_spikes)
        print 'ACTION ', i_action
        reward[counter] = demo22.offline_update(i_state, i_action, block)
        counter +=1



print 'REWARD = ', reward


plt.figure(101)
plt.plot(reward)

plt.figure(102)
plt.plot(np.loadtxt("full_conn_list.dat")[:,2])


plt.show()






