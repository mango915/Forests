import numpy as np

def pairing(configs):
    # In each subplot there are more absent species than present (just an observation)
    # S+ - S-
    S_present = np.count_nonzero(configs, axis = (1)).flatten()
    # S_absent = S - S_present -> S_pm = 2*S_present - S
    S_pm = 2*S_present - S # Broadcasting
    # constraint C_0 = < (S+ - S-)^2 >
    return np.mean(np.power(S_pm,2))


def model_m(configs):
    # assuming configs have a shape of (n, S) with S = 299
    p_i = np.count_nonzero(configs, axis = 0)/configs.shape[0]
    return 2*p_i - 1


def compute_energy(configs, L_multipliers):
    """Computes the energy of a configuration."""
    model_pair = pairing(configs)
    model_m_i = model_m(configs)
    model_parameters = np.concatenate((model_pair[np.newaxis], model_m_i))
    return np.dot(model_parameters, L_multipliers)


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

    def __init__ (self, lagrange_multipliers, exp_constraints, S, M = 100,
                  N = 1000, max_acceptance = 0.1):
        self.S = S # spin glass dimension
        self.M = M # acceptance check interval
        self.N = N # Number of samples after condition reached
        self.max_acceptance = max_acceptance
        self.L_multipliers = lagrange_multipliers
        self.exp_constraints = exp_constraints
        self.acceptance_history = []
        self.model_configs = None


    def dE(self, Sp, spin, k):
        l_k = self.L_multipliers[k]
        l_0 = self.L_multipliers[0]
        return 2*l_k*spin + 4*l_0*spin*(2*Sp - self.S + spin) # <----------- VALUES? ( > eps, < 0 ?)


    def acceptance(self, Sp, spin, k):
        """Implements Metropolis choice."""
        # regularizer?
        dE = self.dE(Sp, spin, k)
        if  dE < 0:
            return True
        else:
            P = np.random.random()
            if P < np.exp(-dE):                 #in teoria no <------------------- BETA?
                return True
            else:
                return False


    def calibrate(self):
        """Starts from a random configurations and sets the first configuration in configs as the one at
           which the acceptance rate of the last M configurations is under max_acceptance."""
        # M <- ricombinations, N <- sampling
        acceptance_rate = 1
        configuration = np.random.choice([+1,-1], size = self.S)

        while(acceptance_rate > self.max_acceptance):
            Sp = np.count_nonzero(configuration+1)

            # sort M indexes between 0 and S to flips
            flip_spins = np.random.randint(low = 0, high = self.S, size = self.M)

            for index in flip_spins:
                spin = -configuration[index]
                if self.acceptance(Sp, spin, index):     # <-------------- acceptances?
                    self.acceptance_history.append(1) #
                    configuration[index] = spin
                    Sp += spin # trick step

            acceptance_rate = len(self.acceptance_history)/self.M
            self.acceptance_history = []

        self.model_configs[0] = configuration
        return configuration

    def sample(self, N = None):
        """Computes and returns N configurations of the system."""
        if N != None:
            self.N = N
        self.model_configs = np.zeros((self.N,self.S))
        configuration = self.calibrate()
        Sp = np.count_nonzero(configuration+1)

        # at the end of calibration the first configuration is already stored,
        # thus we have to sample other N-1 configurations

        # sort M indexes between 0 and S to flips
        flip_spins = np.random.randint(low = 0, high = self.S, size = self.N-1)

        for k, index in enumerate(flip_spins):
            spin = -configuration[index]
            if self.acceptance(Sp, spin, index):
                configuration[index] = spin
                Sp += spin
                # if the new config is chosen, we memorize it
                self.model_configs[k + 1] = configuration
            else:
                # otherwise we store another time the last config
                self.model_configs[k + 1] = configuration

        return self.model_configs
