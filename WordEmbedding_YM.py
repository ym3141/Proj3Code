import numpy as np
import os.path

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical
from DataProcessing import encodedShake, loadShake_char, Convert2SonnetNaive
from RNNProcessing import build_model_LSTM, gen_lines_word2vec
from keras.callbacks import EarlyStopping, ModelCheckpoint


def generateTrainData(encodedSonnets, dim=10):
    '''
    Generate training data from sonnets
    '''
    seqs = []
    # Iterate over each document in reference text
    for sonnet in encodedSonnets:
        # Add each input sequence with corresponding output character
        for i in range(dim, len(sonnet)):
            seqs.append(sonnet[i - dim: i + 1])
    seqs = np.array(seqs)
    X_code = seqs[:, :-1]
    y_code = seqs[:, -1]

    return X_code, y_code

def gnerateEmbedTrain(encodedSonnets, window_size=4):
    '''
    Function to generate data for word2vec
    '''
    trainX = []
    trainY = []
    for sonnet in encodedSonnets:
        for idx, word in enumerate(sonnet):
            for jdx in np.arange(- window_size, window_size + 1):
                kdx = idx + jdx
                if kdx >= 0 and kdx < len(sonnet) and jdx != 0:
                    trainX.append(word)
                    trainY.append(sonnet[kdx])
    return np.array(trainX), np.array(trainY)


def trainWord2Vec(latentFactorN = 10, weightSave = './TrainingTemp/Word2vecWeight.npy', epoch=20):
    '''
    Train a word2vec model based on Shakespear's sonnet, and save it's weight in TrainingTemp folder
    '''
    encodedSonnets, encodedSyllaDict, code2word, punc2code = encodedShake()
    X, y = gnerateEmbedTrain(encodedSonnets)    
    
    model = Sequential()
    model.add(Dense(units=latentFactorN, input_dim=len(code2word)))
    model.add(Dense(units=len(code2word), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    model.fit(to_categorical(X), to_categorical(y), epochs=epoch, verbose=1)

    weights = model.layers[0].get_weights()[0]
    weights = weights / np.max(np.abs(weights))
    np.save(weightSave, weights)

    return weights

def most_similar_pairs(weight_matrix, word_to_index):
    N = len(word_to_index)
    simiMat = np.ones((N, N)) * 100
    for i in np.arange(0, N):
        for j in np.arange(i+1, N):
            simiMat[i, j] = np.sum((weight_matrix[i] - weight_matrix[j])**2)
    
    return simiMat

def trainRNNvec(encodedSonnets, weight, modelSave='./TrainingTemp/RNN_word2vec-LSTM.h5'):
    '''
    Train a RNN model based on word2vec weight matrix
    '''

    trainX, trainY = generateTrainData(encodedSonnets)
    trainXvec = []
    trainYvec = []

    for i in range(len(trainX)):
        print(trainX[i], trainY[i])
        trainYvec.append(weight[trainY[i]])
        seqX = []
        for x in trainX[i]:
            seqX.append(weight[x])
        trainXvec.append(seqX)

    trainXvec = np.array(trainXvec)
    trainYvec = np.array(trainYvec)

    model = build_model_LSTM(trainXvec.shape[1:], trainYvec.shape[1])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    model_checkpoint = ModelCheckpoint(modelSave, monitor='val_loss', save_best_only=True)

    # Compile and fit model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(trainXvec, trainYvec, batch_size=64, epochs=100, validation_split=0.2, callbacks=[early_stopping, model_checkpoint], verbose=1)

    return model


if __name__ == '__main__':
    encodedSonnets, encodedSyllaDict, code2word, punc2code = encodedShake()
    

    if os.path.isfile('./TrainingTemp/Word2vecWeight.npy'):
        weights = np.load('./TrainingTemp/Word2vecWeight.npy')
    else:
        weights = trainWord2Vec(latentFactorN=50)

    if os.path.isfile('./TrainingTemp/RNN_word2vec-LSTM.h5'):
        model = load_model('./TrainingTemp/RNN_word2vec-LSTM.h5')
    else:
        model = trainRNNvec(encodedSonnets, weights)

    word2code = dict([(code2word[i], i) for i in code2word])
    seed_text = 'shall i compare thee to a summer\'s day ? \n thou art more lovely and more temperate \n'
    seed_code = []
    for word in seed_text.split(' '):
        seed_code.append(word2code[word])
    
    lines = gen_lines_word2vec(model, seed_code, weights, lineNum=14)

    print(Convert2SonnetNaive(lines, code2word)[1])
        
    pass