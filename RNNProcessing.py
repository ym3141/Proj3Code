import numpy as np
import string

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM


def text2seq(text, seq_len=40, seq_int=1):
    '''
    Convert reference text to training sequences of fixed length.
    
    Inputs:
    - text: iterable of string(s) representing reference corpus
    - seq_len: length of sequences to be used as inputs to the model
    - seq_int: interval at which to sample subsequences from the text, where
        seq_int=1 corresponds to taking all possible subsequences
    
    Outputs:
    - X_code: array of shape (n_seq, seq_len) representing n_seq input sequences
        from the text, with each character encoded as its integer label
    - y_code: array of shape (n_seq, 1) representing n_seq output characters
        from the text, with each character encoded as its integer label
    - char2code: dict where char2code[char] is integer label for character char
    - code2char: dict where code2char[code] is character for integer label code
    '''
    seqs = []
    # Iterate over each document in reference text
    for doc in text:
        # Add each input sequence with corresponding output character
        for i in range(seq_len, len(doc), seq_int):
            seqs.append(doc[i - seq_len:i + 1])
    
    # Map characters to integer codes
    char2code = {}
    code2char = {}
    for code, char in enumerate(sorted(set(''.join(text)))):
        char2code[char] = code
        code2char[code] = char
    
    # Convert characters of training data into integer codes
    seqs_code = np.array([[char2code[char] for char in seq] for seq in seqs])
    
    # Split sequences into inputs and outputs
    X_code = seqs_code[:, :-1]
    y_code = seqs_code[:, -1]
    
    return X_code, y_code, char2code, code2char


def seq2cat(X_code, y_code, n_char):
    '''
    Use one-hot encoding to convert integer classes to categorical data.
    
    Inputs:
    - X_code: array of shape (n_seq, seq_len) representing n_seq input sequences
        from the text, with each character encoded as its integer label
    - y_code: array of shape (n_seq, 1) representing n_seq output characters
        from the text, with each character encoded as its integer label
    - n_char: number of characters represented
    
    Outputs:
    - X: array of shape (n_seq, seq_len, n_char) representing n_seq inputs
        from the text, using one-hot encoding of integer labels
    - y: array of shape (n_seq, n_char) representing n_seq outputs
        from the text, using one-hot encoding of integer labels
    '''
    X = keras.utils.to_categorical(X_code, num_classes=n_char)
    y = keras.utils.to_categorical(y_code, num_classes=n_char)
    
    return X, y


def build_model_LSTM(input_shape, output_shape, n_layers=1, n_units=[128]):
    '''
    Construct character-based LSTM model with specified architecture of LSTM
    layers followed by fully connected output layer with softmax activation.
    
    Inputs:
    - input_shape: input shape for first layer
    - output_shape: output shape for last layer
    - n_layers: integer specifying number of LSTM layers
    - n_units: iterable of length n_layers specifying number of units per layer
    
    Output:
    - model: Keras model representing specified architecture
    '''
    model = Sequential()
    model.add(LSTM(n_units[0], input_shape=input_shape,
                   return_sequences=(n_layers > 1)))
    for layer in range(1, n_layers):
        model.add(LSTM(n_units[layer],
                       return_sequences=(n_layers > layer + 1)))
    model.add(Dense(output_shape, activation='softmax'))
    
    return model


def sample_code(probs, T=1):
    '''
    Return sample for integer codes given probabilities of each class and
    desired temperature.
    
    Inputs:
    - probs: probabilities of each class
    - T: temperature for softmax sampling, controlling variance in samples
    
    Output:
    - code: integer code corresponding to sampled class
    '''
    # Compute temperature-adjusted probabilities
    probs_T = probs.squeeze() ** (1 / T)
    probs_T /= np.sum(probs_T)
    
    # Draw sample
    code = np.random.choice(len(probs_T), p=probs_T)
    
    return code


def sampleCloseWeight(vec, weights):
    '''
    Function to select a word based on how close it is to the predicted word2vec vector

    Input:
    -vec: predicted vector of the word
    - 
    '''

    dists = []
    for wordVec in weights:
        dists.append(np.sum(np.abs(wordVec-vec)))

    dists = np.array(dists)
    # choices = dists.argpartition(10)[0:10]
    revDists = (1 / dists) ** 7
    revDists = revDists / revDists.sum()

    return np.random.choice(len(revDists), p = revDists)
        

def gen_chars(model, seed, code2char, n_chars=1000, T=1,
              verbose=True):
    '''
    Generates text using provided model and seed text.
    
    Inputs:
    - model: trained Keras model
    - seed: initial input sequence
    - code2char: dict where code2char[code] is character for integer label code
    - n_chars: number of characters to generate
    - T: temperature for softmax sampling, controlling variance in samples
    - verbose: flag indicating whether to print input and output sequences
    
    Output:
    - seq_out: string with n_chars characters generated by model
    '''
    if verbose:
        print('Seed Sequence:')
        print(''.join([code2char[code] for code in seed]))
        print('')
    
    seq_in = seed
    seq_out = ['' for i in range(n_chars)]
    
    # Generate each character
    for i in range(n_chars):
        # Convert input sequence to categorical encoding
        seq_in_cat = keras.utils.to_categorical(seq_in,
                                                num_classes=len(code2char))
        
        # Sample character based on predicted class probabilities
        probs = model.predict(seq_in_cat[np.newaxis])
        code = sample_code(probs, T)
        seq_out[i] = code2char[code]
        
        # Update input sequence with new character
        seq_in = np.concatenate((seq_in[1:], [code]))
    
    seq_out = ''.join(seq_out)
    if verbose:
        print('Output Sequence:')
        print(seq_out)
    
    return seq_out


def gen_lines(model, seed, code2char, n_lines=14, T=1, verbose=True):
    '''
    Generates text using provided model and seed text.
    
    Inputs:
    - model: trained Keras model
    - seed: initial input sequence
    - code2char: dict where code2char[code] is character for integer label code
    - n_lines: number of lines to generate
    - T: temperature for softmax sampling, controlling variance in samples
    - verbose: flag indicating whether to print input and output sequences
    
    Output:
    - seq_out: string with n_lines lines generated by model
    '''
    if verbose:
        print('Seed Sequence:')
        print(''.join([code2char[char] for char in seed]))
        print('')
    
    seq_in = seed
    seq_out = ''
    
    # Generate characters until desired number of lines is reached
    lines = 1
    while lines <= n_lines:
        # Convert input sequence to categorical encoding
        seq_in_cat = keras.utils.to_categorical(seq_in,
                                                num_classes=len(code2char))
        
        # Sample character based on predicted class probabilities
        probs = model.predict(seq_in_cat[np.newaxis])
        code = sample_code(probs, T)
        char = code2char[code]
        seq_out += char
        
        # Update input sequence with new character
        seq_in = np.concatenate((seq_in[1:], [code]))
        
        # Update number of lines if needed
        if char == '\n':
            lines += 1
    
    seq_out = seq_out[:-1]
    if verbose:
        print('Output Sequence:')
        print(seq_out)
    
    return seq_out


def gen_lines_word2vec(model, seed, weights, lineNum=14):
    '''
    Generates text using provided model and seed text (word embedding version).
    
    Inputs:
    - model: trained Keras model for word2vec input
    - seed: initial input sequence
    - weights: word2vec weight matrix
    - n_lines: number of lines to generate
    
    Output:
    - seqOutput: a list of ji
    '''

    seqInput = []
    for code in seed[-10:]:
        seqInput.append(weights[code])
    seqInput = np.array(seqInput)
    seqOutput = []
    
    # Generate characters until desired number of lines is reached
    lines = 1
    newlineFlag = True
    while lines <= lineNum: 
        # Sample character based on predicted class probabilities
        nextVec = model.predict(seqInput[np.newaxis])
        code = sampleCloseWeight(nextVec, weights)

        if not (newlineFlag and code == 3209):
            seqOutput.append(code)
            newlineFlag = False
        
        # Update input sequence with new character
        seqInput = np.concatenate((seqInput[1:], [weights[code]]))
        if code == 3209:
            if not newlineFlag:
                lines += 1
            newlineFlag = True

    return seqOutput


def gen_lines_sylls(model, seed, code2char, syllableDic,
                    n_lines=14, n_sylls=10,
                    T=1, verbose=True):
    '''
    Generates text using provided model and seed text.
    
    Inputs:
    - model: trained Keras model
    - seed: initial input sequence
    - code2char: dict where code2char[code] is character for integer label code
    - syllableDic: dict mapping words to numbers of syllables
    - n_lines: number of lines to generate
    - n_sylls: number(s) of syllables per line (used cyclically)
    - T: temperature for softmax sampling, controlling variance in samples
    - verbose: flag indicating whether to print input and output sequences
    
    Output:
    - seq_out: string of given line and syllable structure generated by model
    '''
    if verbose:
        print('Seed Sequence:')
        print(''.join([code2char[char] for char in seed]))
        print('')
    
    seq_in = seed
    seq_out = ''
    
    # Set number of syllables per line if needed
    if not hasattr(n_sylls, "__len__"):
        n_sylls = [n_sylls]
    if len(n_sylls) < n_lines:
        n_sylls_all = np.tile(n_sylls,
                              np.ceil(n_lines / len(n_sylls)).astype(int))
        n_sylls_all = n_sylls_all[:n_lines]
    else:
        n_sylls_all = n_sylls
    
    # Generate characters until desired number of lines is reached
    lines = 1
    while lines <= np.max((n_lines, len(n_sylls_all))):
        # Generate words until desired number of syllables is reached
        word = ''
        sylls = 0
        gen_words = True
        
        while gen_words:
            # Convert input sequence to categorical encoding
            seq_in_cat = keras.utils.to_categorical(seq_in,
                                                    num_classes=len(code2char))

            # Sample character based on predicted class probabilities
            probs = model.predict(seq_in_cat[np.newaxis])
            code = sample_code(probs, T)
            char = code2char[code]
            word += char

            # Update input sequence with new character
            seq_in = np.concatenate((seq_in[1:], [code]))
            
            # Check if word is in dictionary
            if char in string.whitespace:
                # Remove leading and trailing whitespace or punctuation
                word = word.strip(string.whitespace + string.punctuation)
                if word in syllableDic:
                    syll = int(syllableDic[word][-1])
                    # Add word if within appropriate number of syllables
                    if (sylls + syll <= n_sylls_all[lines-1]):
                        seq_out = seq_out + word + ' '
                        sylls += syll
                    
                    # Break line if appropriate number of syllables reached
                    if sylls == n_sylls_all[lines-1]:
                        seq_out = seq_out[:-1] + '\n'
                        gen_words = False
                        lines += 1
                word = ''
    
    seq_out = seq_out[:-1]
    if verbose:
        print('Output Sequence:')
        print(seq_out)
    
    return seq_out