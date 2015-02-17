import demo22
from parameters import *
import simplified_bg_pynn as network
from matplotlib import pyplot as plt
import numpy as np



################
# INIT STORAGE #
################

reward = np.zeros(n_blocks*n_trials)
w_d1 = np.zeros(n_blocks*n_trials) 
w_d2 = np.zeros(n_blocks*n_trials) 


################
#    RUN       #
################


#demo22.init_weights(n_states, m_actions)
counter = 0
demo22.init_weights()
for block in xrange(n_blocks):
    for trial in xrange(n_trials):
        i_state = (trial+block) % n_states
        print 'TRIAL ', trial, 'BLOCK ', block
        gpi_spikes = network.run(i_state)
        print 'GPI SPIKES ', gpi_spikes
        i_action = np.argmax(gpi_spikes)
        ## save
        reward[counter] = demo22.offline_update(i_state, i_action, block)
        w_d1[counter] = np.mean(np.loadtxt("D1_state0_to_action0.dat"))
        w_d2[counter] = np.mean(np.loadtxt("D2_state0_to_action0.dat"))
        print 'STATE', i_state, 'ACTION ', i_action, 'REWARD ', reward[counter]
        counter +=1



print 'REWARD = ', reward

plt.figure(101)
plt.subplot(331)
plt.plot(reward)

#plt.figure(102)
plt.subplot(332)
plt.plot(np.loadtxt("full_conn_list.dat")[:,2])

#plt.figure(103)
plt.subplot(333)
plt.imshow(np.loadtxt("full_conn_list.dat")[:,:2])

plt.subplot(334)
plt.plot(w_d1)
plt.subplot(335)
plt.plot(w_d2)
plt.show()






