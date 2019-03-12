import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from DataProcessing import encodedShake, loadShake_char


def generateTrainData(encodedSonnets, dictLen, dim=10):
    seqs = []
    # Iterate over each document in reference text
    for sonnet in encodedSonnets:
        # Add each input sequence with corresponding output character
        for i in range(dim, len(sonnet)):
            seqs.append(sonnet[i - dim: i + 1])
    seqs = np.array(seqs)
    X_code = to_categorical(seqs[:, :-1], num_classes=dictLen)
    y_code = to_categorical(seqs[:, -1], num_classes=dictLen)

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

    encodedSonnets, encodedSyllaDict, code2word, punc2code = encodedShake()
    X, y = gnerateEmbedTrain(encodedSonnets)    
    
    model = Sequential()
    model.add(Dense(units=latentFactorN, input_dim=len(code2word)))
    model.add(Dense(units=len(code2word), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    model.fit(to_categorical(X), to_categorical(y), epochs=epoch, verbose=1)

    weights = model.layers[0].get_weights()[0]
    np.save(weightSave, weights)

    return weights

def most_similar_pairs(weight_matrix, word_to_index):
    N = len(word_to_index)
    simiMat = np.ones((N, N)) * 100
    for i in np.arange(0, N):
        for j in np.arange(i+1, N):
            simiMat[i, j] = np.sum((weight_matrix[i] - weight_matrix[j])**2)
    
    return simiMat


if __name__ == '__main__':
    encodedSonnets, encodedSyllaDict, code2word, punc2code = encodedShake()
    # weight = trainWord2Vec()
    weight = np.load('./TrainingTemp/Word2vecWeight.npy')
    pairs = most_similar_pairs(weight, code2word)
    pass