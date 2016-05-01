# Imports.
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from SwingyMonkey import SwingyMonkey

#setup global vars
SCREEN_WIDTH  = 600
SCREEN_HEIGHT = 400
BINSIZE = 50 
GAMMA = 0.9 
VELSTATES = 5 
EPSF = 0.0001
BINSIZE = 50 


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch = 1

        self.Q = np.zeros((2,SCREEN_WIDTH/BINSIZE+1,SCREEN_HEIGHT/BINSIZE+1,VELSTATES))
        self.k = np.zeros((2,SCREEN_WIDTH/BINSIZE+1,SCREEN_HEIGHT/BINSIZE+1,VELSTATES)) # number of times action a has been taken from state s
        self.iters = 0
        self.m = [0, 0]
        self.scores = []
        self.best_score = 50
        self.bestQ = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        #increment the epoch on reset
        self.epoch += 1

    def action_callback(self, state):
        self.iters += 1
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        ################
        # set up current state 
        ################

        # distance to the next tree trunk
        dis = state['tree']['dist'] / BINSIZE

        #velocity 
        vel=state['monkey']['vel'] / BINSIZE
        
        #height of bottom of next tree trunk
        htt = (state['tree']['top']-state['monkey']['top']+0) / BINSIZE
        
        if np.abs(vel) > 2:
            vel = np.sign(vel)*2
        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        #define action function that return 0 or 1 randomly.  
        def action_do(prob=0.5):
            #return 1 if npr.rand() < p else 0
            return 1 if npr.rand() < prob else 0   
            
        def default_action(p=0.5):
            return 1 if npr.rand() < p else 0     
            
        ################
        # set up previous state 
        ################
            
        #as long as we have a last action, use the previous state info
        # to update Q matrix 
        new_action = default_action()
        if not self.last_action == None:
            
            # figure out the values of the previous state
            dislast = self.last_state['tree']['dist'] / BINSIZE
            httlast = (self.last_state['tree']['top']-self.last_state['monkey']['top']+0) / BINSIZE
            vellast = self.last_state['monkey']['vel'] / BINSIZE

            if np.abs(vellast) > 2:
                vellast = np.sign(vellast)*2
            
            #find the maximum Q value for use in the Q-function
            max_Q = np.max(self.Q[:,dis,htt,vel])
            
            #if the new Q value is higher, then jump.  Otherwise do nothing
            new_action = 1 if self.Q[1][dis,htt,vel] > self.Q[0][dis,htt,vel] else 0
            
            # setup epsilon
            if self.k[new_action][dis,htt,vel] > 0:
                eps = EPSF/self.k[new_action][dis,htt,vel]
            else:
                eps = EPSF
            if (npr.rand() < eps):
                new_action = default_action()

            ALPHA = 1/self.k[self.last_action][dislast,httlast,vellast]
            self.Q[self.last_action][dislast,httlast,vellast] += ALPHA*(self.last_reward+GAMMA*max_Q-self.Q[self.last_action][dislast,httlast,vellast])

        self.m[0] = state['monkey']['top']

        self.last_action = new_action
        self.last_state  = state
        self.k[new_action][dis,htt,vel] += 1
        return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward




def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(round(swing.score, 1))

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 150, 1) 


	# Save history. 
	np.savetxt('hist_eps_0001.csv',hist,delimiter=",")
