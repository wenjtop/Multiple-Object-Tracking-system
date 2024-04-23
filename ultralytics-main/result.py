import pandas as pd

df = pd.read_csv('results.csv')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
cols = 3
figsize = (20, 25)
fig1 = plt.figure(num=1, figsize=figsize)
plt.rcParams.update({"font.size":20})
ax = []
gs = gridspec.GridSpec( (len(df.keys())-4)// cols + 1, cols)
gs.update(hspace=0.35)
gs.update(bottom=0.001)
gs.update(top=0.96)

gs.update(left=0.05)
gs.update(right=0.97)
keys = []
for i in df.keys()[1:-3]:
    keys.append(i)
keys = keys[:3]+keys[7:10]+keys[3:5]+keys[6:7]
# keys3 = keys[3]
# keys4 = keys[4]
# keys[3:8] = keys[5:]
# keys[8] = keys3
# keys[9] = keys4
xlabel = ['a','b','c','d','e','f','g','h','i','j']
for i, key in enumerate(keys):
    row = (i // cols)
    col = i % cols
    ax.append(fig1.add_subplot(gs[row, col]))
    ax[-1].set_title(str(key).replace(' ', ''))
    ax[-1].plot(np.array(list(df[df.keys()[0]])), np.array(list(df[key])), 'o', ls='-', ms=8, color='#2ca02c')
    ax[-1].set_xlabel('epoch\n('+xlabel[i]+')')
plt.savefig('1.png')
plt.show()

#

