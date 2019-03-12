import numpy as np

from DataProcessing import *
from RNNProcessing import *

from keras.models import load_model


if __name__ == '__main__':
    # Load data
    sonnets, syllableDic = loadShake_char(stripPunc=False)
    X_code, y_code, char2code, code2char = text2seq(sonnets)
    X, y = seq2cat(X_code, y_code, len(char2code))

    # Load model
    model = load_model('RNN_char-LSTM.h5')

    # Choose seed
    seed_text = 'shall i compare thee to a summer\'s day?\n'
    seed = np.array([char2code[char] for char in seed_text])
    print('Seed Sequence:\n%s' % seed_text)

    # Generate haiku as 3-line sequence with appropriate syllable counts
    T_all = [0.25, 0.75, 1.5]
    for T in T_all:
        seq_out = gen_lines_sylls(model, seed, code2char, syllableDic,
                                  n_lines=3, n_sylls=[5, 7, 5],
                                  T=T, verbose=False)
        print('\nHaiku (Temperature %.2f):' % T)
        print('-' * 60)
        print(seq_out)
        print('-' * 60)
