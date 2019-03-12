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
    seed = X_code[np.random.choice(len(X_code))]
    print('Seed Sequence:')
    print(''.join([code2char[code] for code in seed]))
    print('')

    # Generate sonnet as 14-line sequence
    T = 0.5
    seq_out = gen_lines(model, seed, code2char, n_lines=14, T=T,
                        verbose=False)
    print('Sonnet (Temperature %.1f):' % T)
    print('-' * 60)
    print(seq_out)
    print('-' * 60)
