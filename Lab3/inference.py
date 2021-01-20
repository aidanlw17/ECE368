import numpy as np
import graphics
import rover
import time

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    #forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    
    # computing the transition matrix p(zn|zn-1)
    transition_matrix = rover.Distribution()
    
    # computing the observation matrix p(x,y | z)
    observation_matrix = rover.Distribution()
    
    for state in all_possible_hidden_states:
        transition_matrix[state] = transition_model(state)
        
        observation_matrix[state] = observation_model(state)
        
        
    # TODO: Compute the forward messages   
    
    
        
    initial_post = observation_matrix[ ( (observations[0]) + ("stay",) ) ]
    
    result = rover.Distribution()
    
    for state in initial_post:
        result[ (state) + ("stay",) ] = initial_post[state] 
         

    forward_messages[0] = result
                
    for i in range(1, num_time_steps):
                
        # p(x,y | zn)
        
        
        forward_messages[i] = rover.Distribution()
        
        # loop through all states possible
                
        # i is zn and i-1 is zn-1
        
        # get the possible states of zn
        for zn in all_possible_hidden_states:
            
            
                        
            result = 0
            
            inner_term = 0
            
            # loop through zn-1 values
            for zn1 in forward_messages[i-1]:
                
                # probability p(z_n | z_n-1)
                probZZ = transition_matrix[zn1]
                
                # check if in the p(zn|zn-1)
                if zn not in probZZ:
                    continue
                
                if forward_messages[i-1][zn1] == 0:
                    continue
                
            
                # summation term of forward message
                inner_term += probZZ[zn] * forward_messages[i-1][zn1]
            
                                      
            # once gone through all values of z_n-1, multiply by p(x|z_n)
            post_xy = observation_model(zn)
            
            if (observations[i] == None):
                probxy = 1
            else:
                if observations[i] not in post_xy:
                    continue
                probxy = post_xy[observations[i]]
            
            
            
            result = probxy * inner_term
            

            if (result != 0):
                forward_messages[i][zn] = result
                
        forward_messages[i].renormalize()
        
     
    
    # backward part
    
    backward_messages[-1] = rover.Distribution()

                
    # start from the rightmost (zN)
    for i in range(num_time_steps-1, 0, -1):
                
        # p(x,y | zn)
        
        
        backward_messages[i-1] = rover.Distribution()
        
        # loop through all states possible
        
        if i == num_time_steps-1:
        
            # initialize backward_messages
            for state in all_possible_hidden_states:
                backward_messages[i][state] = 1
        
        # i is zn and i-1 is zn-1
        
        # get the possible states of zn-1 
        for state in all_possible_hidden_states:
            
            # probability p(z_n | z_n-1)
            probZZ = transition_matrix[state]
            
            #print("probZZ", probZZ)
            
            result = 0
            
            for zn in backward_messages[i]:
                
                                
                # check if in the p(z|z)
                if zn not in probZZ:
                    continue
                
                if backward_messages[i][zn] == 0:
                    continue
                
                
                inner_term = probZZ[zn] * backward_messages[i][zn]
                
                                          
                # once gone through all values of z_n-1, multiply by p(x|z)
                post_xy = observation_model(zn)
                
                if (observations[i] == None):
                    probxy = 1
                else:
                    if observations[i] not in post_xy:
                        continue
                    probxy = post_xy[observations[i]]
                
                
                
                result += probxy * inner_term
            

            if (result != 0):
                backward_messages[i-1][state] = result
                
        backward_messages[i-1].renormalize() 
            
      
        
                
    for i in range(0,num_time_steps):
        marginals[i] = rover.Distribution()
        for state in all_possible_hidden_states:   
            
            if forward_messages[i][state]*backward_messages[i][state] != 0 :
                marginals[i][state] = forward_messages[i][state]*backward_messages[i][state]
                
        marginals[i].renormalize()
            
    # TODO: Compute the marginals 
            
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    
    num_time_steps = len(observations)
    w = [None] * num_time_steps
    phi = [None] * (num_time_steps-1)
    estimated_hidden_states = [None] * num_time_steps
    
    # computing the transition matrix p(zn|zn-1)
    transition_matrix = rover.Distribution()
    
    # computing the observation matrix p(x,y | z)
    observation_matrix = rover.Distribution()
    
    for state in all_possible_hidden_states:
        transition_matrix[state] = transition_model(state)
        
        observation_matrix[state] = observation_model(state)

    # initial w(z1)
    initial_post = observation_matrix[ ( (observations[0]) + ("stay",) ) ]
    
    result = rover.Distribution()
    
    for state in initial_post:
        result[ (state) + ("stay",) ] = np.log(initial_post[state] * prior_distribution[state + ("stay",)])
         

    w[0] = result
    
        
    
    # loop
    for i in range(1, num_time_steps):
                
        w[i] = rover.Distribution()
        
        phi[i-1] = rover.Distribution()
        
        # loop through all states possible
                
        # i is zn and i-1 is zn-1
        
        # get the possible states of zn
        for zn in all_possible_hidden_states:
                        
            # set to large negative number
            inner_term = -90000000
            
            # loop through zn-1 values
            for zn1 in w[i-1]:
                
                # probability p(z_n | z_n-1)
                probZZ = transition_matrix[zn1]
                
                # check if in the p(zn|zn-1)
                if zn not in probZZ:
                    continue
                
                if w[i-1][zn1] == 0:
                    continue

            
                # max term of forward message                
                if (np.log(probZZ[zn]) + w[i-1][zn1] > inner_term):
                    inner_term = np.log(probZZ[zn]) + w[i-1][zn1]
                    phi[i-1][zn] = zn1
            
                                      
            # once gone through all values of z_n-1, multiply by p(x|z_n)
            post_xy = observation_model(zn)
            
            if (observations[i] == None):
                probxy = 1
            else:
                if observations[i] not in post_xy:
                    continue
                probxy = post_xy[observations[i]]
            
            
            
            result = np.log(probxy) + inner_term
            
            if (inner_term != -90000000):
                w[i][zn] = result
                
    # find max z_n
    maximum = -90000
    arg_max = None
    
    estimated_hidden_states[-1] = rover.Distribution()
    
    for key in w[-1]:
        if w[-1][key] > maximum:
            arg_max = key
            maximum = w[-1][key]
        
    estimated_hidden_states[-1] = arg_max
    

    # backtrack
    for i in range(num_time_steps-2, -1, -1):
        estimated_hidden_states[i] = rover.Distribution()
        
        estimated_hidden_states[i] = phi[i][arg_max]
        
        # next term used to backtrack
        arg_max = phi[i][arg_max]
        
                
    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = False
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')
    
     
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    

    
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
        
        
    # my code
    print(marginals[30])
    
    
    # PART 4
    # Get the argmax(zi) of p(z|x,y)
    
    max_prob = [None] * num_time_steps
    
    for i in range(0, num_time_steps):
        
        maximum = -90000
        arg_max = None
        
        # get the armgax
        for state in marginals[i]:
            if marginals[i][state] > maximum:
                maximum = marginals[i][state]
                arg_max = state
                
        max_prob[i] = arg_max
        
    # find the error term for each 
    p_vert = 0
    p_max = 0
    
    
    for i in range(0, num_time_steps):
        
        if estimated_states[i] == hidden_states[i] :
            p_vert += 1
            
        if max_prob[i] == hidden_states[i] :
            p_max += 1
            
    p_vert = 1 - p_vert/100
    p_max = 1 - p_max/100
    
    print(p_vert, p_max)
    
    print(max_prob)
    
    
    
    
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
