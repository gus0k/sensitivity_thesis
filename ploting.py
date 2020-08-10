import dill
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.ticker as mtick
from pathlib import Path

# from matplotlib.backends.backend_pgf import FigureCanvasPgf
# mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
mpl.use('pgf')

def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])

plt.rcParams.update({
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
})

### True Data / Error Data

# with open('stats_sensitivity_rolling', 'rb') as fh:
#     rolling = dill.load(fh)

receding_selling = []
receding_keeping = []
for fl in Path('results/').glob('*receding*'):
    with open(fl, 'rb') as fh:
        d = dill.load(fh)
        receding_selling.append(d[0][0])
        receding_keeping.append(d[1][0])

rolling_selling = []
rolling_keeping = []
for fl in Path('results/').glob('*rolling*'):
    with open(fl, 'rb') as fh:
        d = dill.load(fh)
        rolling_selling.append(d[0][0])
        rolling_keeping.append(d[1][0])


datasets = list(map( np.array, [rolling_keeping, rolling_selling, receding_keeping]))
names = [
        'Rolling Horizon (Keeping)',
        'Rolling Horizon (Selling)',
        'Receding Horizon',]

errors = [100 * ((dt[:, 1] / dt[:, 0]) - 1) for dt in datasets] 

df = pd.DataFrame(errors)
df = df.transpose()
df.columns = names


### Ploting receding horizon

receding  = np.array([(r[1] - r[0]) / np.abs(r[0]) for r in receding_selling]) * 100
roll_keep = np.array([(r[1] - r[0]) / np.abs(r[0]) for r in rolling_keeping]) * 100
roll_sell = np.array([(r[1] - r[0]) / np.abs(r[0]) for r in rolling_selling])  * 100

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0] = sns.boxplot(receding,  ax=ax[0], showfliers=False)
ax[0].xaxis.set_major_formatter(mtick.PercentFormatter())
ax[0].set_title(names[2])
ax[1] = sns.boxplot(roll_keep, ax=ax[1], showfliers=False)
ax[1].set_title(names[0])
ax[2] = sns.boxplot(roll_sell, ax=ax[2], showfliers=False)
ax[2].set_title(names[1])
fig.show()
fig.savefig('sensitivitiy_boxplots.pgf')

#### Parameters for the plot
#n_bins = 50

####

#fig, ax = plt.subplots()
#for i in range(3):
#    ax.hist(errors[i],
#            n_bins,
#            histtype='step',
#            density=True,
#            cumulative=True,
#            label=names[i])
#ax.legend()
##ax.set_xlim([-1, 1])
#fix_hist_step_vertical_line_at_end(ax)
#fig.show()

#data = pd.melt(df)
#fig, ax = plt.subplots()
#ax = sns.boxplot(x='variable', y='value', data=data)
#fig.show()

