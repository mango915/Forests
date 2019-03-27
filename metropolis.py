import numpy as np

class Metropolis:
    """
    Metropolis algorithm for an Ising model with the Hamiltonian given by maximum entropy principle.
    
    Parameters
    ------------
    
    S              : number of spins of a configuration (size of the system)
    M              : number of configurations used to compute the running rate of acceptance (memory)
    N              : number of configurations to be sampled
    max_acceptance : acceptance rate under which starts the importance sampling (es. 0.10)
    
    Attributes
    ------------
    
    L_multipliers    : list or numpy array of m Lagrange multipliers 
    constraint_funcs : list of m functions of the spins of a configuration, to implement the max-ent constraints
    energy           : the Hamiltonian for a configuration of spins is the scalar product between the lagrangian
                        multipliers and the constraint functions computed for the given configuration
    configs          : numpy ndarray of N rows and S columns, to store the sampled configurations
    history          : list of the records of all the accepted and refused steps 

    """
    
    def __init__ (self, S, M = 100, N = 1000, max_acceptance = 0.1):
        self.S = S
        self.M = M
        self.N = N
        self.max_acceptance = max_acceptance
        self.history = []
        self.L_multipliers = None
        self.constraint_funcs = None
        self.configs = None
        
        
    def set_hamiltonian(self, L_multipliers, constraint_funcs):
        """ Sets L_multipliers and constraint_funcs."""
        self.L_multipliers = np.array(L_multipliers)
        self.constraint_funcs = constraint_funcs
        if len(L_multipliers) != (len(constraint_funcs)):
            raise Exception("L_multipliers and constraint_funcs must have the same length!")
        
    def compute_energy(self, s): # should I use self?
        """Computes the energy of a configuration."""
        constraints =  np.array(list(map(lambda f: f(s), self.constraint_funcs)))
        return np.dot(constraints, self.L_multipliers)

    
    def choice(self, last_config, new_config): # should I use self? 
        """Implements Metropolis choice."""
        last_energy = self.compute_energy(last_config)
        new_energy = self.compute_energy(new_config)
        if  new_energy < last_energy:
            return True
        else:
            P = np.random.random()
            if P < np.exp(last_energy-new_energy):
                return True
            else:
                return False
 
    def calibrate(self):
        """Starts from a random configurations and sets the first configuration in configs as the one at 
           which the acceptance rate of the last M configurations is under max_acceptance."""
        # starting configuration
        s_last = np.random.choice([+1,-1], size = self.S)
        # sort an index between 0 and S to flip
        flip_spin = np.random.randint(low = 0, high = self.S)
        # the next configuration (still to be accepted) is similar to the last
        s_next = np.copy(s_last)
        # except for the flipped spin
        s_next[flip_spin] = -s_last[flip_spin]
        if self.choice(s_last, s_next): #returns True or False, if True we accept it
            self.history.append(1) # keep the record
            s_last = np.copy(s_next)
        else:
            self.history.append(0)
            
        # we want to do at least M runs and we want the last M runs 
        # to have an average acceptance rate below max_acceptance
        # accept = +1, refuse = 0, history.mean() = accept/total
        
        index = min(len(self.history), self.M)
        
        while ( len(self.history) < self.M ) or \
               ( (np.mean(self.history[-index:]) > self.max_acceptance)  and  (len(self.history) >= self.M) ):
            flip_spin = np.random.randint(low = 0, high = self.S)
            s_next = np.copy(s_last)
            s_next[flip_spin] = -s_last[flip_spin]

            if self.choice(s_last, s_next): 
                self.history.append(1)
                s_last = np.copy(s_next)
            else:
                self.history.append(0)
                
            # number of elements (from the last one) among which compute the average
            index = min(len(self.history), self.M)
        
        #print("Acceptance rate criteria satisfied after %.d attempts."%len(self.history))
        #print("History of calibration: ", self.history)
        #print("Mean acceptance rate: ", np.mean(self.history[-index:])) 
        self.configs[0] = np.copy(s_last)
        
    def sample(self, N = None):
        """Computes and returns N configurations of the system."""
        if N != None:
            self.N = N
        self.configs = np.zeros((self.N,self.S))
        self.calibrate() 
        
        # at the end of calibration the first configuration is already stored, 
        # thus we have to sample other N-1 configurations
        for i in range(self.N-1):
            s_last = self.configs[i] 
            flip_spin = np.random.randint(low = 0, high = self.S)
            s_next = np.copy(s_last)
            s_next[flip_spin] = -1*s_last[flip_spin]

            if self.choice(s_last, s_next): 
                self.history.append(1)
                self.configs[i+1] = s_next # if the new config is chosen, we memorize it
            else:
                self.history.append(0)
                self.configs[i+1] = s_last # otherwise we store another time the last config
        
        return self.configs
        
