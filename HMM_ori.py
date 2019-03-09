#########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np

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
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


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

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2A)
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

        ###
        ###
        ###

        #max_seq = ''
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
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2Bi)
        for i in range(self.L):
            alphas[1][i] = self.A_start[i] * self.O[i][x[0]]

        for t in range(2,M+1):
            for i in range(self.L):
                prob = 0
                for j in range(self.L):
                    temp = alphas[t-1][j] * self.A[j][i] * self.O[i][x[t-1]]
                    prob = prob + temp
                alphas[t][i] = prob
            if normalize:
                norm = sum(alphas[t])
                for j in range(self.L):
                    alphas[t][j] = 1.0*alphas[t][j]/norm

        ###
        ###
        ###

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
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2Bii)
        for i in range(self.L):
            betas[M][i] = 1

        for t in range(M-1,-1,-1):
            for i in range(self.L):
                prob = 0
                for j in range(self.L):
                    temp = betas[t+1][j] * self.A[i][j] * self.O[j][x[t]]
                    prob = prob + temp
                betas[t][i] = prob
            if normalize:
                norm = sum(betas[t])
                for j in range(self.L):
                    betas[t][j] = 1.0*betas[t][j]/norm
        ###
        ###
        ###

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

        # Calculate each element of A using the M-step formulas.

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2C)
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

        ###
        ###
        ###

        # Calculate each element of O using the M-step formulas.

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2C)
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
        ###
        ###
        ###

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

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2D)

        for itera in range(N_iters):
            print("Iteration: " + str(itera+1))

            # Numerator and denominator for updating A and O.
            A_n = [[0. for _ in range(self.L)] for _ in range(self.L)] # sum of probabilites of transitions from state i to state j
            O_n = [[0. for _ in range(self.D)] for _ in range(self.L)] # sum of probabilites in state i and observing j
            A_d = [0. for _ in range(self.L)] # sum of probabilites of transitions from state i
            O_d = [0. for _ in range(self.L)] # sum of probabilites in state i

            # For each row of input sequence:
            for t1 in range(len(X)):

                # Sequence length
                M = len(X[t1])

                # Compute the alphas and betas
                alphas = self.forward(X[t1], normalize=True)
                betas = self.backward(X[t1], normalize=True)

                for t2 in range(1, M + 1):
                    p1 = [0. for _ in range(self.L)]

                    # E step - Compute p1
                    norm_p1 = 0
                    for q in range(self.L):
                        p1[q] = alphas[t2][q] * betas[t2][q]
                        norm_p1 = norm_p1 + p1[q]

                    # Normalize p1.
                    for q in range(self.L):
                        p1[q] = p1[q]/norm_p1

                    # Update O_d, O_n and A_d
                    for q in range(self.L):
                        O_d[q] = O_d[q] + p1[q]
                        O_n[q][X[t1][t2-1]] += p1[q]
                        if t2 < M:
                            A_d[q] += p1[q]

                for t2 in range(1, M):
                    p2 = [[0. for _ in range(self.L)] for _ in range(self.L)]

                    # E step - Compute p2
                    norm_p2 = 0
                    for qi in range(self.L):
                        for qj in range(self.L):
                            p2[qi][qj] = alphas[t2][qi] * self.A[qi][qj] * self.O[qj][X[t1][t2]] * betas[t2+1][qj]
                            norm_p2 = norm_p2 + p2[qi][qj]

                    # Normalize p2.
                    for qi in range(self.L):
                        for qj in range(self.L):
                            p2[qi][qj] /= norm_p2

                    # Update A_n
                    for qi in range(self.L):
                        for qj in range(self.L):
                            A_n[qi][qj] += p2[qi][qj]

            # M - step
            for qi in range(self.L):
                for qj in range(self.L):
                    self.A[qi][qj] = A_n[qi][qj] / A_d[qi]

            for qi in range(self.L):
                for oj in range(self.D):
                    self.O[qi][oj] = O_n[qi][oj] / O_d[qi]

        ###
        ###
        ###

        pass


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

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2F)

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

        ###
        ###
        ###

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

    return HMM