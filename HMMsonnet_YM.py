from HMM_fast import HiddenMarkovModel, unsupervised_HMM
from DataProcessing import encodedShake, code2sonnet


if __name__ == '__main__':
    encodedSonnets, encodedSyllaDict, code2word, punc2code = encodedShake()

    HMmodel = unsupervised_HMM(encodedSonnets, 40, 100)

    print(' '.join(code2sonnet(encodedSonnets[0], code2word)))

    sonnet = HMmodel.generate_emission(200)[0]
    
    print(' '.join(code2sonnet(sonnet, code2word)))
    print()
