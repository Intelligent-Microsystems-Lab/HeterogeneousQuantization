# IMSL Lab - University of Notre Dame
# Author: Zephan M Enciso, Jake Leporte, Clemens JS Schaefer
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd


# Competitor Performance.
competitors = {

    'pact_resnet18': {
        # https://arxiv.org/pdf/1805.06085.pdf
        'eval_err': np.array([1 - 0.644, 1 - 0.681, 1 - 0.692,
                              1 - 0.698]) * 100,
        'size_mb': np.array([2, 3, 4, 5
                             ]) * 11679912 / 8_000_000 + 0.03840000 / 2,
        'name': 'PACT ResNet18*',
        'alpha': .25,
        # no first and last layer quant
    },

    'dsq_resnet18': {
        # https://arxiv.org/abs/1908.05033
        'eval_err': np.array([1 - 0.6517, 1 - 0.6866, 1 - 0.6956]) * 100,
        'size_mb': np.array([2, 3, 4
                             ]) * 11679912 / 8_000_000 + 0.03840000 / 2,
        'name': 'DSQ ResNet18*',
        'alpha': .25,
    },

    'lsq_resnet18': {
        # https://arxiv.org/abs/1902.08153
        'eval_err': np.array([1 - 0.676, 1 - 0.702, 1 - 0.711,
                              1 - 0.711]) * 100,
        'size_mb': np.array([2, 3, 4, 8
                             ]) * 11679912 / 8_000_000 + 0.03840000 / 2,
        'name': 'LSQ ResNet18*',
        'alpha': .25,
    },

    'lsqp_resnet18': {
        # https://arxiv.org/abs/2004.09576
        # they claim to be a natural extension... so also first and last?
        'eval_err': np.array([1 - 0.668, 1 - 0.694, 1 - 0.708]) * 100,
        'size_mb': np.array([2, 3, 4]) * 11679912 / 8_000_000 + 0.03840000 / 2,
        'name': 'LSQ+ ResNet18*',
        'alpha': .25,
    },


    'ewgs_resnet18': {
        # https://arxiv.org/abs/2104.00903
        'eval_err': np.array([1 - 0.553, 1 - 0.67, 1 - 0.697,
                              1 - 0.706]) * 100,
        'size_mb': np.array([1, 2, 3, 4
                             ]) * 11679912 / 8_000_000 + 0.03840000 / 2,
        'name': 'EWGS ResNet18*',
        'alpha': .25,
    },


    'psgd_resnet18': {
        # https://arxiv.org/pdf/2005.11035.pdf
        'eval_err': np.array([1 - 0.6345, 1 - 0.6951, 1 - 0.7013, ]) * 100,
        'size_mb': np.array([4, 6, 8]) * 11679912 / 8_000_000 + 0.03840000 / 2,
        'name': 'PSGD ResNet18',
        'alpha': .25,
    },

    'qil_resnet18': {
        # "We did not quantize the first and the last layers as was done in..."
        # https://arxiv.org/abs/1808.05779
        'eval_err': np.array([1 - 0.657, 1 - 0.692, 1 - 0.701, 1 - 0.704,
                              ]) * 100,
        'size_mb': np.array([2, 3, 4, 5
                             ]) * 11679912 / 8_000_000 + 0.03840000 / 2,
        'name': 'QIL ResNet18*',
        'alpha': .25,
    },

    'mixed_resnet18': {
        # activation budget 380 KB against
        # https://arxiv.org/abs/1905.11452
        'eval_err': np.array([0.2992]) * 100,
        'size_mb': np.array([5.4]),  # max act 0.38MB
        'max_act_mb': np.array([.38]),
        'name': 'Mixed ResNet18',
        'alpha': .25,
    },


}


def inefficient_frontier(file, x_axis="Weight Size", y_axis="Error",
                         summing=None):
  df = pd.read_csv(file)
  df['Error'] = 1 - df['Accuracy']
  if summing:
    df['Sum'] = sum(df[item] for item in summing)
  df = df.sort_values(x_axis)
  frontier = []
  for index, row in df.iterrows():
    if not len(frontier) or row[y_axis] <= df.at[frontier[-1], y_axis]:
      frontier.append(index)
  return [list(thing) for thing in (df.iloc[frontier][x_axis] / 1000,
          df.iloc[frontier][y_axis],
          df.iloc[~df.index.isin(frontier)][x_axis] / 1000,
          df.iloc[~df.index.isin(frontier)][y_axis])]


resnet_mixed = inefficient_frontier(
    'figures/resnet18_mixed.csv', x_axis='Sum', y_axis="Error",
    summing=['Act Size Max', 'Weight Size'])
resnet_mixed_gran = inefficient_frontier(
    'figures/resnet18_mixed_gran.csv', x_axis='Sum', y_axis="Error",
    summing=['Act Size Max', 'Weight Size'])
resnet_mixed_sur = inefficient_frontier(
    'figures/resnet18_mixed_sur.csv', x_axis='Sum', y_axis="Error",
    summing=['Act Size Max', 'Weight Size'])
resnet_mixed_sur_gran = inefficient_frontier(
    'figures/resnet18_mixed_sur_gran.csv', x_axis='Sum', y_axis="Error",
    summing=['Act Size Max', 'Weight Size'])


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'sans'
mpl.rcParams['mathtext.it'] = 'sans:bold'
mpl.rcParams['mathtext.default'] = 'bf'


font_size = 23
plt.rc('font', family='Helvetica', weight='bold')
fig, ax = plt.subplots(figsize=(16.5, 8.5))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
ax.yaxis.set_tick_params(width=5, length=10, labelsize=font_size)

for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(5)

for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontweight('bold')


ax.plot(resnet_mixed[0], resnet_mixed[1], marker='x', label='Mixed',
        ms=20, markeredgewidth=5, linewidth=5, color='green')
ax.scatter(resnet_mixed[2], resnet_mixed[3], marker='x',
           s=20**2, linewidth=5, color='green', alpha=.25)


ax.plot(resnet_mixed_gran[0], resnet_mixed_gran[1], marker='x',
        label='Mixed Granular', ms=20, markeredgewidth=5, linewidth=5,
        color='red')
ax.scatter(resnet_mixed_gran[2], resnet_mixed_gran[3],
           marker='x', s=20**2, linewidth=5, color='red', alpha=.25)


ax.plot(resnet_mixed_sur[0], resnet_mixed_sur[1], marker='x',
        label='Mixed Surrogate', ms=20, markeredgewidth=5, linewidth=5,
        color='blue')
ax.scatter(resnet_mixed_sur[2], resnet_mixed_sur[3],
           marker='x', s=20**2, linewidth=5, color='blue', alpha=.25)


ax.plot(resnet_mixed_sur_gran[0], resnet_mixed_sur_gran[1], marker='x',
        label='Mixed Granular Surrogate', ms=20, markeredgewidth=5,
        linewidth=5, color='magenta')
ax.scatter(resnet_mixed_sur_gran[2], resnet_mixed_sur_gran[3],
           marker='x', s=20**2, linewidth=5, color='magenta', alpha=.25)


ax.set_xscale('log')
plt.xticks([3, 4, 5, 6, ], [
    '3', '4', '5', '6'])
ax.set_xlabel("Max Activation Size + Weight Size (MB)",
              fontsize=font_size, fontweight='bold')
ax.set_ylabel("Eval Error (%)", fontsize=font_size, fontweight='bold')
plt.legend(
    bbox_to_anchor=(-.05, 1.02, 1.05, 0.2),
    loc="lower left",
    ncol=3,
    mode="expand",
    borderaxespad=0,
    frameon=False,
    prop={'weight': 'bold', 'size': font_size}
)
plt.tight_layout()
plt.savefig('figures/overview.png', dpi=300)
plt.close()
