from HMM_fast import HiddenMarkovModel, unsupervised_HMM
from DataProcessing import encodedShake, Convert2SonnetNaive


if __name__ == '__main__':
    encodedSonnets, encodedSyllaDict, code2word, punc2code = encodedShake()

    HMmodel = unsupervised_HMM(encodedSonnets, 40, 50)

    print(Convert2SonnetNaive(encodedSonnets[0], code2word)[0])
    print(Convert2SonnetNaive(encodedSonnets[0], code2word)[1])

    sonnet = HMmodel.generate_emission(200)[0]
    
    print(Convert2SonnetNaive(sonnet, code2word)[1])
