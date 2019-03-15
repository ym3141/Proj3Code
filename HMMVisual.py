#%%
import numpy as np
import matplotlib.pyplot as plt
from DataProcessing import encodedShake, Convert2SonnetNaive

encodedSonnets, encodedSyllaDict, code2word, punc2code = encodedShake()

HMMa = np.load('./TrainingTemp/HMM.A.npy')
HMMo = np.load('./TrainingTemp/HMM.O.npy')

#%%
import seaborn as sns

sns.set_style("white")
fig, ax = plt.subplots(1, 1, facecolor='w')
fig.dpi=300
transMap = ax.imshow(HMMa)
fig.colorbar(transMap)

#%%
maxI, maxO = (int(HMMa.argmax()/40), HMMa.argmax()%40)
print(int(HMMa.argmax()/40), HMMa.argmax()%40)

#%%
from wordcloud import WordCloud

stateN = 15

wcDict = {}
for i in range(len(code2word) - 1):
    wcDict[code2word[i]] = HMMo[stateN, i]
wcDict['\\n'] = HMMo[stateN, -1]

wc = WordCloud(background_color="white", max_words=1000)
wc.generate_from_frequencies(wcDict)

plt.imshow(wc)
plt.title('State {0}'.format(stateN))

#%%
max5 = HMMa.flatten().argpartition(1600-5)[-5:]
maxI, maxO = ((max5/40).astype(int), max5%40)