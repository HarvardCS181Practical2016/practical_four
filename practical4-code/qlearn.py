import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from SwingyMonkey import SwingyMonkey

# Instantiate our global variables. Use this to experiment with differing bin-sizes and gamma values 
width = 600
height = 400
bin_size = 50 
gamma = 0.9 
vel_zero = 5 
discount_factor = 0.0001

class Learner(object):

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch = 1
        self.iters = 0

        self.Q = np.zeros((2,width/bin_size+1,height/bin_size+1,vel_zero))
        self.k = np.zeros((2,width/bin_size+1,height/bin_size+1,vel_zero)) 

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

        # Get the current distance to the next tree, the current velocity and the height of the next tree trunk
        # Standardize based on bin size
        dis = state['tree']['dist']/bin_size 
        vel=state['monkey']['vel']/bin_size 
        if np.abs(vel) > 2:
            vel = np.sign(vel)*2
        htt = (state['tree']['top']-state['monkey']['top'])/bin_size
              
        if npr.rand() < 0.5:
            new_action  = 1
        else:
            new_action = 0


        if not self.last_action == None:
            
            #  Figure out the values of the previous state
            dislast = self.last_state['tree']['dist']/bin_size
            vellast = self.last_state['monkey']['vel']/bin_size
            if np.abs(vellast) > 2:
                vellast = np.sign(vellast)*2
            httlast = (self.last_state['tree']['top']-self.last_state['monkey']['top']+0)/bin_size

            # Take the previous state information, plug into the Q function and solve
            max_Q = np.max(self.Q[:,dis,htt,vel])
            
            # If the new Q value is higher, then jump.  Otherwise do nothing
            if self.Q[1][dis,htt,vel] > self.Q[0][dis,htt,vel]:
                new_action = 1
            else:
                new_action =0
            
            # Calculate epsilon
            if self.k[new_action][dis,htt,vel] > 0:
                eps = discount_factor/self.k[new_action][dis,htt,vel]
            else:
                eps = discount_factor

            ''' Uncomment out this section if we want to use our epsilon value to be used in determining our action! '''    
            #test = npr.rand()
            #if (npr.rand() < eps):
             #   if test < 0.5:
              #      new_action = 1
               # else:
                #    new_action = 0

            # Calculate alpha, update our q matrix    
            alpha = 1/self.k[self.last_action][dislast,httlast,vellast]
            self.Q[self.last_action][dislast,httlast,vellast] += alpha*(self.last_reward+gamma*max_Q-self.Q[self.last_action][dislast,httlast,vellast])

        # Update and return actions    
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