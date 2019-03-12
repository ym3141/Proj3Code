# import keras
# from keras.utils import to_categorical
from DataProcessing import encodedShake, loadShake_char


# def generate_onehot_dict(word_list):
#     """
#     Takes a list of the words in a text file, returning a dictionary mapping
#     words to their index in a one-hot-encoded representation of the words.
#     """
#     word_to_index = {}
#     i = 0
#     for word in word_list:
#         if word not in word_to_index:
#             word_to_index[word] = i
#             i += 1
#     return word_to_index

# # def word2vec()

def generateTrainData(encodedSonnets, dictLen, dim=10):
    seqs = []
    # Iterate over each document in reference text
    for sonnet in encodedSonnets:
        # Add each input sequence with corresponding output character
        for i in range(dim, len(sonnet)):
            seqs.append(sonnet[i - seq_len: i + 1])
    
    X_code = to_categorical(seqs[:, :-1], num_classes=dictLen)
    y_code = to_categorical(seqs[:, -1], num_classes=dictLen)

    return X_code, y_code




if __name__ == '__main__':
    encodedSonnets, encodedSyllaDict, code2word, punc2code = encodedShake()
    X, y = generateTrainData(encodedSonnets, len(code2word), dim=10)    
    
    pass