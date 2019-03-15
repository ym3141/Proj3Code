import numpy as np

from DataProcessing import *
from RNNProcessing import *

from keras.models import load_model

def gen_haikus(model_file, char2code, code2char, syllableDic):
    '''
    Load Keras model saved in [model_file] and generate haikus
    (poems with 3 lines of 5, 7, and 5 syllables) for specified
    seed and temperatures, using mappings between characters and codes
    given in [char2code] and [code2char] as well as syllable dictionary
    given in [syllableDic].
    '''
    # Load model
    model = load_model(model_file)

    # Choose seed
    seed_text = 'shall i compare thee to a summer\'s day?\n'
    seed = np.array([char2code[char] for char in seed_text])
    print('Seed Sequence:\n%s' % seed_text)

    # Set temperatures to consider
    T_all = [0.25, 0.75, 1.5]
    for T in T_all:
        # Generate haiku as 3-line sequence with appropriate syllable counts
        seq_out = gen_lines_sylls(model, seed, code2char, syllableDic,
                                  n_lines=3, n_sylls=[5, 7, 5],
                                  T=T, verbose=False)
        print('\nHaiku (Temperature %.2f):' % T)
        print('-' * 60)
        print(seq_out)
        print('-' * 60)


if __name__ == '__main__':
    # Load data
    sonnets, syllableDic = loadShake_char(stripPunc=False)
    X_code, y_code, char2code, code2char = text2seq(sonnets)
    X, y = seq2cat(X_code, y_code, len(char2code))

    # Generate haikus based on line and syllable number
    # Original architecture, optimized on validation loss
    print('=' * 60)
    print('128 LSTM Units (minimum val_loss)')
    print('=' * 60)
    gen_haikus('TrainingTemp/RNN_char-LSTM.h5',
               char2code, code2char, syllableDic)
    # Optimized achitecture, optimized on validation loss
    print('=' * 60)
    print('256 LSTM Units (minimum val_loss)')
    print('=' * 60)
    gen_haikus('TrainingTemp/RNN_char-LSTM_valid.h5',
               char2code, code2char, syllableDic)
    # Optimized achitecture, optimized on training loss
    print('=' * 60)
    print('256 LSTM Units (minimum loss)')
    print('=' * 60)
    gen_haikus('TrainingTemp/RNN_char-LSTM_train.h5',
               char2code, code2char, syllableDic)
