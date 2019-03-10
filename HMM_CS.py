########################################
# CS/CNS/EE 155 2018-2019
# Problem Set 6
#
# Author:       Christina Su
# Description:  Set 6 solution code
#
# Adapted from Set 6 skeleton code
# Author:       Andrew Kang
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of Set 6. Once each part is implemented, you can simply
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

import numpy as np
import random

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
        self.A_start = 1 / self.L * np.ones(self.L)


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
        probs = np.zeros((M + 1, self.L))
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        ### TODO: Insert Your Code Here (2A)
        # Initialize probabilities and sequences for prefixes of length 1
        probs[1] = self.A_start * self.O[:, x[0]]
        seqs[1] = [str(s) for s in range(self.L)]

        # Consider prefixes of increasing length
        for i in range(2, M + 1):
            # Consider possible end states for prefixes of length i
            for j in range(self.L):
                # Find probabilities of ending in given state for
                # possible end states of prefixes of length (i - 1)
                probs_j = probs[i - 1] * self.A[:, j] * self.O[j, x[i - 1]]

                # Extract maximum probability and corresponding prefix
                probs[i][j] = np.max(probs_j)
                seqs[i][j] = seqs[i - 1][np.argmax(probs_j)] + str(j)

        # Find state sequence of maximum probability
        max_seq = seqs[M][np.argmax(probs[M])]
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
        alphas = np.ones((M + 1, self.L))

        ### TODO: Insert Your Code Here (2Bi)
        # Initialize probabilities for prefixes of length 1
        alphas[1] = self.A_start * self.O[:, x[0]]

        # Consider prefixes of increasing length
        for i in range(2, M + 1):
            # Find probabilities of ending in given state for
            # possible end states of prefixes of length (i - 1)
            alphas[i] = (alphas[i - 1][np.newaxis] @ self.A) * \
                        self.O[:, x[i - 1]]

            # Normalize probabilities if specified
            if normalize:
                alphas[i] /= np.sum(alphas[i])

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
        betas = np.ones((M + 1, self.L))

        ### TODO: Insert Your Code Here (2Bii)
        # Initialize probabilities for suffixes after M observations
        betas[M] = 1

        # Consider suffixes of increasing length
        for i in range(M - 1, 0, -1):
            # Find probabilities of observing given suffix for
            # possible hidden state sequences starting at y^(i + 1)
            betas[i] = (betas[i + 1] * self.O[:, x[i]])[np.newaxis] @ \
                       self.A.T

            # Normalize probabilities if specified
            if normalize:
                betas[i] /= np.sum(betas[i])

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

        # Calculate each element of A and O using the M-step formulas

        ### TODO: Insert Your Code Here (2C)
        # Initialize variables to track numerators and denominators of
        # each element in A and O matrices
        A_num = np.zeros((self.L, self.L))
        A_den = np.zeros(self.L)
        O_num = np.zeros((self.L, self.D))
        O_den = np.zeros(self.L)

        # Traverse input and state sequences to update counts
        for i in range(len(X)):
            for j in range(len(X[i])):
                # Update counts for elements of transition matrix A
                if j > 0:
                    A_num[Y[i][j - 1], Y[i][j]] += 1
                    A_den[Y[i][j - 1]] += 1

                # Update counts for elements of observation matrix O
                O_num[Y[i][j]][X[i][j]] += 1
                O_den[Y[i][j]] += 1

        # Compute A and O
        self.A = A_num / A_den[:, np.newaxis]
        self.O = O_num / O_den[:, np.newaxis]


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

        ### TODO: Insert Your Code Here (2F)
        # Sample initial state
        states.append(random.choices(range(self.L), weights=self.A_start)[0])

        # Iterate for specified length of sequence
        for i in range(M):
            # Sample emission given current hidden state
            emission.append(random.choices(range(self.D),
                                           weights=self.O[states[-1]])[0])

            # Sample transition given current hidden state
            states.append(random.choices(range(self.L),
                                         weights=self.A[states[-1]])[0])

        # Extract hidden states corresponding to emission
        states = states[:-1]

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

    # Randomly initialize and normalize matrix A.
    random.seed(2019)
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

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
