from DataProcessing import encodedShake, code2sonnet
import nltk
from nltk.corpus import cmudict

d = cmudict.dict()

encodedSonnets, encodedSyllaDict, code2word, punc2code = encodedShake()

def Vowel_num(pron):
    syl = ""
    for elem in pron:
        if "1" in elem or "2" in elem: syl += "1"
        elif "0" in elem: syl += "0"
    return syl

def Rhyme(pron):
    rhy = ""
    v = 0
    for i in range(len(pron) - 1, -1, -1):
        if "1" in pron[i] or "2" in pron[i] or "0" in pron[i]: 
            v = i
            break
    for j in range(v, len(pron)):
        rhy += pron[j]
    return rhy

rhyme_dic = {} # dictionary of rhyme
syllable_dic = {} # dictionary of syllable (e.g. 1, 0, 10, ...)

for i in range(len(code2word)):
    single_word = code2word[i]
    single_word_syl = []
    single_word_rhyme = []
    if single_word in d:
        for pron in d[single_word]:
            single_word_syl.append(Vowel_num(pron)) 
            single_word_rhyme.append(Rhyme(pron)) 
        for s in single_word_syl:
            if s not in syllable_dic:
                syllable_dic[s] = []
            if single_word not in syllable_dic[s]:
                syllable_dic[s].append(single_word)
        for r in single_word_rhyme:
            if r not in rhyme_dic:
                rhyme_dic[r] = []
            if single_word not in rhyme_dic[r]:
                rhyme_dic[r].append(single_word)
