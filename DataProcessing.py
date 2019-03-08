import numpy as ny
import re

sonnetNumLineRe = re.compile(r'\s+\d+\n')
wordPuncRe = re.compile(r"[\w']+|[.,!?;]")
wordOnlyRe = re.compile(r"[\w']+")


# Function to load data of Shakespear to a list of sonnets.
# Each sonnet is further seperated into lines
# Each line is further seperated into words and punctruations unless specified (then punctruations will be ignored)
# Word like "Feed'st" and "youth's" will be treated as one single word
def loadShake(sepPuncs=True):
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
                    newSonnet.append(splitLine)

        sonnets.append(newSonnet)
        sonnets = sonnets[1:]
    
    return sonnets


if __name__ == '__main__':
    sonnets = loadShake(False)
    pass
