import numpy as ny
import re

sonnetNumLineRe = re.compile(r'\s+\d+\n')
wordPuncRe = re.compile(r"[\w'-]+|[.,!?;]")
wordOnlyRe = re.compile(r"[\w'-]+")


# Function to load data of Shakespear to a list of sonnets.
# Each sonnet is further seperated into lines
# Each line is further seperated into words and punctruations unless specified (then punctruations will be ignored)
# Word like "Feed'st" and "youth's" will be treated as one single word
# Hyphened words are also treated as one word
# Single quotes not in the middle of the word will be ignored
def loadShake(sepPuncs=True, ignoreSingleQuotes=False):
    sonnets = []
    with open('./data/shakespeare.txt', 'r') as f:
        newSonnet = []
        for line in iter(lambda: f.readline(), ''):
            if sonnetNumLineRe.match(line):
                sonnets.append(newSonnet)
                newSonnet = []
            else:
                if not line == '\n':
                    if sepPuncs:
                        splitLine = wordPuncRe.findall(line)
                    else:
                        splitLine = wordOnlyRe.findall(line)

                    if ignoreSingleQuotes:
                        for idx, word in enumerate(splitLine):
                            newWord = word
                            if word.endswith('\''):
                                newWord = newWord[0:-1]
                                splitLine[idx] = newWord
                            elif word.startswith('\''):
                                newWord = newWord[1:]
                                splitLine[idx] = newWord
                    splitLine = [x for x in splitLine if x]
                    newSonnet.append(splitLine)

        sonnets.append(newSonnet)
        sonnets = sonnets[1:]
    
    return sonnets


# function to return a dict of the syllable dict
# key will be word, and value will be a tuple that contains several possible number of syllables
# note values in tuples are all str, since we have to deal with things like "E1"
# example: 
# {"'gainst": ('1',), "'greeing": ('E1', '2'), "'scaped": ('1',) ... } 
def loadSyllable():
    syllDict = dict()
    with open('./data/Syllable_dictionary.txt') as f:
        for line in iter(lambda: f.readline(), ''):
            splitLine = line.split()
            syllDict[splitLine[0]] = tuple(splitLine[1:])
    return syllDict


if __name__ == '__main__':
    # check if the every word loaded from Shakespear can be found in the syllable dict 
    syllDict = loadSyllable()
    for sonnet in loadShake(False):
        for line in sonnet:
            for word in line:
                if not word.lower() in syllDict:
                    print(word) 
    pass
