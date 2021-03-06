import re
import string

sonnetNumLineRe = re.compile(r'\s+\d+\n')
wordPuncRe = re.compile(r"[\w'-]+|[.,!?;]")
wordOnlyRe = re.compile(r"[\w'-]+")

puncSet = list('!,.;?\n')


def loadSyllable():
    # Function to return a dict of the syllable dictionary
        # Key will be word, and value will be a tuple that contains several possible number of syllables
        # Note values in tuples are all str, since we have to deal with things like "E1"
        # Example: 
        # {"'gainst": ('1',), "'greeing": ('E1', '2'), "'scaped": ('1',) ... } 
    syllDict = dict()
    with open('./data/Syllable_dictionary.txt') as f:
        for line in iter(lambda: f.readline(), ''):
            splitLine = line.split()
            syllDict[splitLine[0]] = tuple(splitLine[1:])
    return syllDict



def loadShake(sepPuncs=True, matchSyllableDic=True, toLower=True):
    # Function to load sonnets and syllable dictionary of Shakespeare
        # For sonnets 
            # Each sonnet is further separated into lines
            # Each line is further separated into words and punctuation unless specified (then punctuation will be ignored)
            # Turn on "matchSyllableDic" to make all words have a syllable dictionary entry (some messy work)
            # toLower will turn everything to lower case 
        # For syllable dict:
            # Key will be word, and value will be a tuple that contains several possible number of syllables
            # Note values in tuples are all str, since we have to deal with things like "E1" 
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



def loadShake_char(stripPunc=True, toLower=True, matchSyllableDic=True):
    # Function to load Shakespearean sonnets and syllable dictionary for
        # character-based model
        # sonnets: list in which each element represents a sonnet
        # stripPunc=True: remove punctuation (except in words) and newline tokens
        # toLower=True: convert to lower case
        # matchSyllableDic=True: ensure that all words are in syllable dictionary
        # (does not apply if stripPunc=False)
        # syllableDic: dict representing syllable dictionary
    # See loadSyllable() for full documentation
    sonnets = []
    syllableDic = loadSyllable()
    
    # Read in data
    with open('./data/shakespeare.txt', 'r') as f:
        sonnet = ''
        for line in iter(lambda: f.readline(), ''):
            # Check for beginning of sonnet
            if sonnetNumLineRe.match(line):
                sonnets.append(sonnet)
                sonnet = ''
            
            # Process nonempty lines
            elif line != '\n':
                # Append full line if punctuation/spacing are not stripped
                if not stripPunc:
                    if toLower:
                        sonnet += line.lower()
                    else:
                        sonnet += line
                    continue
                    
                # Split line into words
                splitLine = wordPuncRe.findall(line)
                
                # Check that all words are in syllable dictionary if specified
                if matchSyllableDic:
                    for idx, word in enumerate(splitLine):
                        # Strip punctuation
                        if word in string.punctuation:
                            splitLine[idx] = ''
                        elif not word.lower() in syllableDic:
                            if word[:-1].lower() in syllableDic:
                                splitLine[idx] = word[:-1]
                            elif word[1:].lower() in syllableDic:
                                splitLine[idx] = word[1:]
                            elif word[1:-1].lower() in syllableDic:
                                splitLine[idx] = word[1:-1]
                            else:
                                print('Word', word, 'rejected.')
                                splitLine[idx] = ''
                
                # Add line stripped of punctuation and newline tokens
                if toLower:
                    splitLine = [word.lower() for word in splitLine if word]
                else:
                    splitLine = [word for word in splitLine if word]
                if sonnet != '':
                    sonnet += ' '
                sonnet += ' '.join(splitLine)
                
        sonnets.append(sonnet)
        sonnets = sonnets[1:]
        
    return sonnets, syllableDic



def encodedShake():
    # Function to load and encode sonnets and syllable dictionary of Shakespeare
        # Also return some useful dictionaries: the dict for number to word/punctuation/newline & the dict for punctuation to number
        # Refer to loadShake comments for more detailed explanation
    sonnets, syllaDict = loadShake()
    wordSetList = []
    for word in [w for sonnet in sonnets for line in sonnet for w in line]:
        if not word in wordSetList:
            wordSetList.append(word)
    wordSetList.append('\n')
    word2code = dict(zip(wordSetList, (range(len(wordSetList)))))

    punc2code = dict()
    for word in word2code:
        if not word in syllaDict:
            punc2code[word] = word2code[word]

    encodedSyllaDict = dict()
    for word in word2code:
        if word in syllaDict:
            encodedSyllaDict[word2code[word]] = syllaDict[word]

    encodedSonnets = []
    for sonnet in sonnets:
        encodedSonnet = []
        for line in sonnet:
            encodedLine = [word2code[w] for w in line]
            encodedSonnet = encodedSonnet + encodedLine + [word2code['\n']]
        encodedSonnets.append(encodedSonnet)

    code2word = dict(zip(range(len(wordSetList)), wordSetList))

    return encodedSonnets, encodedSyllaDict, code2word, punc2code


def Convert2SonnetNaive(codeSeq, code2word):
    '''
    A coverter that convert code sequence (not words) to a sonnet, naively.
    It will break at each '\n' and count to 14 lines.
    '''

    sonnet = []
    linecount = 1
    startOfLine = True
    for code in codeSeq[1:]:
        word = code2word[code]
        if word not in puncSet:
            if startOfLine:
                sonnet = sonnet + [word]
                startOfLine = False
            else:
                sonnet = sonnet + [' ', word]
        elif word == '\n':
            sonnet = sonnet + [word]
            startOfLine = True
            linecount += 1
        else:
            sonnet.append(word)
        
        if linecount == 15: 
            return True, ''.join(sonnet)
    if linecount == 14:
        return True, ''.join(sonnet)
    else:
        return False, ''.join(sonnet)


if __name__ == '__main__':
    # check if every word loaded from Shakespeare can be found in the syllable dict
    sonnets, syllDict = loadShake(sepPuncs=False)
    for sonnet in sonnets:
        for line in sonnet:
            for word in line:
                if not word in syllDict:
                    print(word) 
    
    result = encodedShake()
    pass
