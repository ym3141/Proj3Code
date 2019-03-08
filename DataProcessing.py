import numpy as ny
import re

sonnetNumLine = re.compile(r'\s+\d+\n')

def loadShake():
    sonnets = []
    with open('./data/shakespeare.txt', 'r') as f:
        newSonnet = []
        for line in iter(lambda: f.readline(), ''):
            if sonnetNumLine.match(line):
                sonnets.append(newSonnet)
                newSonnet = []
            else:
                if not line == '\n':
                    newSonnet.append(line)

        sonnets.append(newSonnet)

        sonnets = sonnets[1:]
    
    return sonnets


if __name__ == '__main__':
    sonnets = loadShake()
    pass
