import re

sonnetNumLineRe = re.compile(r'\s+\d+\n')
wordPuncRe = re.compile(r"[\w'-]+|[.,!?;]")
wordOnlyRe = re.compile(r"[\w'-]+")

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


# Function to load sonnets and syllable dictionary of Shakespear.
# For sonnets 
    # Each sonnet is further seperated into lines
    # Each line is further seperated into words and punctruations unless specified (then punctruations will be ignored)
    # Turn on "mathchSyllableDic" to make all word has a syllable dictionary entry (some messy work)
    # toLower will turn everything to lower case 
# For syllable dict:
    # key will be word, and value will be a tuple that contains several possible number of syllables
    # note values in tuples are all str, since we have to deal with things like "E1" 
def loadShake(sepPuncs=True, matchSyllableDic=True, toLower=True):
    syllableDic = loadSyllable()
    sonnets = []
    with open('./data/shakespeare.txt', 'r') as f:
        newSonnet = []
        for line in iter(lambda: f.readline(), ''):
            if sonnetNumLineRe.match(line):
                sonnets.append(newSonnet)
                newSonnet = []
            else:
                if not line == '\n':
                    # take care of flags

                    if sepPuncs:
                        splitLine = wordPuncRe.findall(line)
                    else:
                        splitLine = wordOnlyRe.findall(line)

                    if matchSyllableDic:
                        for idx, word in enumerate(splitLine):
                            if word == '\'':
                                splitLine[idx] = ''
                            if not word.lower() in syllableDic:
                                if word[0:-1].lower() in syllableDic:
                                    splitLine[idx] = word[0:-1]
                                elif word[1: ].lower() in syllableDic:
                                    splitLine[idx] = word[1: ]
                                elif word[1: -1].lower() in syllableDic:
                                    splitLine[idx] = word[1: -1]
                    
                    if toLower:
                        splitLine = [x.lower() for x in splitLine if x]
                    else:
                        splitLine = [x for x in splitLine if x]

                    newSonnet.append(splitLine)

        sonnets.append(newSonnet)
        sonnets = sonnets[1:]
    
    return sonnets, syllableDic


# Function to load and encode sonnets and syllable dictionary of Shakespear.
    # Also return some useful dictionaries: the dict for everything to number & the dict for punctruation to number
    # refers to loadShake comments for more detailed explanation.
def encodedShake():
    sonnets, syllaDict = loadShake()
    wordSetList = []
    for word in [w for sonnet in sonnets for line in sonnet for w in line]:
        if not word in wordSetList:
            wordSetList.append(word)
    word2code = dict(zip(wordSetList, (range(len(wordSetList)))))

    punc2code = dict()
    for word in word2code:
        if not word in syllDict:
            punc2code[word] = word2code[word]

    encodedSyllaDict = dict()
    for word in word2code:
        if word in syllDict:
            encodedSyllaDict[word2code[word]] = syllaDict[word]

    encodedSonnets = []
    for sonnet in sonnets:
        encodedSonnet = []
        for line in sonnet:
            encodedLine = [word2code[w] for w in line]
            encodedSonnet.append(encodedLine)
        encodedSonnets.append(encodedSonnet)


    return encodedSonnets, encodedSyllaDict, word2code, punc2code
    


if __name__ == '__main__':
    # check if the every word loaded from Shakespear can be found in the syllable dict 
    sonnets, syllDict = loadShake(sepPuncs=False)
    for sonnet in sonnets:
        for line in sonnet:
            for word in line:
                if not word in syllDict:
                    print(word) 
    
    result = encodedShake()
    pass
