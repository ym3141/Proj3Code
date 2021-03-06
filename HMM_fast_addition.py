import random
import numpy as np
from time import time

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = np.array(A)
        self.O = np.array(O)
        self.A_start = np.ones(self.L) / self.L


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(self.L):
            probs[1][i] = self.A_start[i] * self.O[i][x[0]]
            seqs[1][i] = str(i)

        for t in range(2,M+1):
            for i in range(self.L):
                prob = 0
                for j in range(self.L):
                    temp = probs[t-1][j] * self.A[j][i] * self.O[i][x[t-1]]
                    if temp > prob:
                        prob = temp
                        seqs[t][i] = seqs[t-1][j] + str(i)
                probs[t][i] = prob

        bestpathprob = 0
        for i in range(self.L):
            if probs[M][i] > bestpathprob:
                bestpathprob  = probs[M][i]
                max_seq = seqs[M][i]


        return max_seq

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = np.zeros((M + 1, self.L))

        for i in range(self.L):
            alphas[1][i] = self.A_start[i] * self.O[i][x[0]]

        for t in range(2, M+1):
            alphas[t] = ((alphas[t-1][np.newaxis] @ self.A) * self.O[:, x[t-1]])[0, :]
            if normalize:
                alphas[t] = alphas[t] / alphas[t].sum()

        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.

        betas = np.zeros((M + 1, self.L))
        betas[M, :] = 1

        for t in range(M-1,-1,-1):
            betas[t] = ((betas[t+1] * self.O[:, x[t]])[np.newaxis] @ self.A.T)[0, :]
            if normalize:
                betas[t] = betas[t] / betas[t].sum()

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        for i in range(self.L):
            for j in range(self.L):
                N_ij = 0 # expected number of transitions from state i to state j
                N_i = 0 # expected number of transitions from state i
                for ii in range(len(Y)):
                    for jj in range(len(Y[ii])-1):
                        if (Y[ii][jj] == i) and (Y[ii][jj+1] == j):
                            N_ij = N_ij + 1
                        if Y[ii][jj] == i:
                            N_i = N_i + 1
                self.A[i][j] = 1.0 * N_ij / N_i


        for i in range(self.L):
            for j in range(self.D):
                N_ij = 0 # expected number of times in state i and observing j
                N_i = 0 # expected number of times in state i
                for ii in range(len(X)):
                    for jj in range(len(X[ii])):
                        if (Y[ii][jj] == i) and (X[ii][jj] == j):
                            N_ij = N_ij + 1
                        if Y[ii][jj] == i:
                            N_i = N_i + 1
                self.O[i][j] = 1.0 * N_ij / N_i

        pass

    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        ### TODO: Insert Your Code Here (2D)
        # Perform specified number of iterations
        for iteration in range(N_iters):
            startTime = time()

            # Initialize variables to track numerators and denominators of
            # each element in A and O matrices
            A_num = np.zeros((self.L, self.L))
            A_den = np.zeros(self.L)
            O_num = np.zeros((self.L, self.D))
            O_den = np.zeros(self.L)

            # Traverse input sequences to compute marginal probabilities
            # using forward-backward algorithm (E step)
            for m in range(len(X)):
                alphas = self.forward(X[m], normalize=True)
                betas = self.backward(X[m], normalize=True)

                # Compute marginal probabilities for hidden states, where
                # y_probs[i][j] denotes probability of state j at position i
                y_probs = alphas * betas
                y_probs[0] = 1
                y_probs /= np.sum(y_probs, axis=1)[:, np.newaxis]

                # Traverse given input sequence
                for n in range(len(X[m])):
                    # Update values for elements of transition matrix A
                    if n > 0:
                        # Compute marginal probabilities for transitions, where
                        # A_probs[j][k] denotes probability of transitioning from
                        # state j at position n to state k at position (n + 1)
                        A_probs = alphas[n][:, np.newaxis] * self.A * \
                                self.O[:, X[m][n]][np.newaxis, :] * \
                                betas[n + 1][np.newaxis, :]
                        A_probs /= np.sum(A_probs)

                        # Update values for elements of transition matrix A
                        A_num += A_probs
                        A_den += y_probs[n]

                    # Update values for elements of observation matrix O
                    O_num[:, X[m][n]] += y_probs[n + 1]
                    O_den += y_probs[n + 1]

            # Compute A and O (M step)
            self.A = A_num / A_den[:, np.newaxis]
            self.O = O_num / O_den[:, np.newaxis]
            print("Iteration: #{0:3d}; Took {1:.2f}s".format(iteration + 1, time() - startTime))


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        state_gen = random.randint(0, self.L-1)

        for i in range(M):
            states.append(state_gen)

            ran = random.uniform(0, 1)
            O_row = self.O[state_gen]
            for j in range(len(O_row)):
                ran = ran - O_row[j]
                if ran <= 1e-6:
                    break
            emission.append(j)

            ran = random.uniform(0, 1)
            A_row = self.A[state_gen]
            for j in range(len(A_row)):
                ran = ran - A_row[j]
                if ran <= 1e-6:
                    break
            state_gen = j

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    random.seed(2019)
    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    #random.seed(2019)
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    #return HMM
    return A,O