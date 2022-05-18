# IMSL Lab - University of Notre Dame
# Author: Zephan M Enciso, Jake Leporte, Clemens JS Schaefer
import matplotlib.pyplot as plt
import matplotlib as mpl

# from scipy.optimize import curve_fit

import numpy as np
import pandas as pd


def convex_hull(points):
  # https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/\
  # Convex_hull/Monotone_chain
  """Computes the convex hull of a set of 2D points.

  Input: an iterable sequence of (x, y) pairs representing the points.
  Output: a list of vertices of the convex hull in counter-clockwise order,
    starting from the vertex with the lexicographically smallest coordinates.
  Implements Andrew's monotone chain algorithm. O(n log n) complexity.
  """

  # Sort the points lexicographically (tuples are compared lexicographically).
  # Remove duplicates to detect the case we have just one unique point.
  points = sorted(set(points))

  # Boring case: no points or a single point, possibly repeated multiple times.
  if len(points) <= 1:
    return points

  # 2D cross product of OA and OB vectors.
  # Returns a positive value, if OAB makes a counter-clockwise turn,
  # negative for clockwise turn, and zero if the points are collinear.
  def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

  # Build lower hull
  lower = []
  for p in points:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
      lower.pop()
    lower.append(p)

  # check for tricky first points
  lower = pd.DataFrame(lower).sort_values(1)

  if lower[0].idxmax() != lower[1].idxmin():
    return lower.drop(lower[0].idxmax()).to_numpy()

  return lower.to_numpy()


def lower_convex_hull(input_file, y_axis="Error", summing=["Weight"]):
  df = pd.read_csv(input_file)
  df['Error'] = 1 - df['Accuracy']
  df['Sum'] = sum(df[item] for item in summing)

  np_base = df[['Error', 'Sum']].to_numpy()
  frontier = convex_hull(list(map(tuple, np_base)))

  idx_not_frontier = set([None if np.sum(
      np_base[x, :] == frontier) else x for x in range(np_base.shape[0])]
  ) - set([None])
  idx_not_frontier = [*idx_not_frontier]

  return [(frontier[:, 1] / 1000).tolist(),
          ((frontier[:, 0] * 100)).tolist(),
          (np_base[idx_not_frontier, 1] / 1000).tolist(),
          ((np_base[idx_not_frontier, 0] * 100)).tolist(),
          ]


resnet_mixed = lower_convex_hull(
    'figures/mbnetv2.csv', y_axis="Error",
    summing=['Act Size Sum', 'Weight Size'])
resnet_mixed_gran = lower_convex_hull(
    'figures/mbnetv2_gran.csv', y_axis="Error",
    summing=['Act Size Sum', 'Weight Size'])
resnet_mixed_sur = lower_convex_hull(
    'figures/mbnetv2_sur.csv', y_axis="Error",
    summing=['Act Size Sum', 'Weight Size'])
resnet_mixed_sur_gran = lower_convex_hull(
    'figures/mbnetv2_gran_sur.csv', y_axis="Error",
    summing=['Act Size Sum', 'Weight Size'])


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

# print(resnet_mixed_sur_gran[0])
# print(resnet_mixed_sur_gran[1])

print(resnet_mixed_sur[0])
print(resnet_mixed_sur[1])
# plt.ylim(31, 38)
ax.set_xscale('log')
plt.xticks([3.0, 4.0, 5.0], [
    '3.0', '4.0', '5.0'])
ax.set_xlabel("Weight Size + Sum Activation Size (MB)",
              fontsize=font_size, fontweight='bold')
ax.set_ylabel("Eval Error (%)", fontsize=font_size, fontweight='bold')
plt.legend(
    bbox_to_anchor=(0., 1.02, 1.05, 0.2),
    loc="lower left",
    ncol=2,
    # mode="expand",
    borderaxespad=0,
    frameon=False,
    prop={'weight': 'bold', 'size': font_size}
)
plt.tight_layout()
plt.savefig('figures/sweeps.png', dpi=300)
plt.close()


def func(x, a, b, c):
  return a * np.exp(-b * x) + c


# Fitting a curve through the points
def fitted_curve(input_file, y_axis="Error", summing=["Weight"]):
  df = pd.read_csv(input_file)
  df['Error'] = 1 - df['Accuracy']
  df['Sum'] = sum(df[item] for item in summing)

  np_base = df[['Error', 'Sum']].to_numpy()

  xp = np.linspace(df['Sum'].min(), df['Sum'].max(), 100)

  z = np.polyfit(np_base[:, 1], np_base[:, 0], 5)
  p = np.poly1d(z)
  frontier = p(xp)

  # popt, pcov = curve_fit(func, np_base[:,1], np_base[:,0])
  # frontier = func(xp, *popt)

  return [(xp / 1000).tolist(),
          (frontier * 100).tolist(),
          (np_base[:, 1] / 1000).tolist(),
          ((np_base[:, 0] * 100)).tolist(),
          ]


resnet_mixed = fitted_curve(
    'figures/mbnetv2.csv', y_axis="Error",
    summing=['Act Size Max', 'Weight Size'])
resnet_mixed_gran = fitted_curve(
    'figures/mbnetv2_gran.csv', y_axis="Error",
    summing=['Act Size Max', 'Weight Size'])
resnet_mixed_sur = fitted_curve(
    'figures/mbnetv2_sur.csv', y_axis="Error",
    summing=['Act Size Max', 'Weight Size'])
resnet_mixed_sur_gran = fitted_curve(
    'figures/mbnetv2_gran_sur.csv', y_axis="Error",
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


ax.plot(resnet_mixed[0], resnet_mixed[1], label='Mixed',
        ms=20, markeredgewidth=5, linewidth=5, color='green')
ax.scatter(resnet_mixed[2], resnet_mixed[3], marker='x',
           s=20**2, linewidth=5, color='green', alpha=.25)


ax.plot(resnet_mixed_gran[0], resnet_mixed_gran[1],
        label='Mixed Granular', ms=20, markeredgewidth=5, linewidth=5,
        color='red')
ax.scatter(resnet_mixed_gran[2], resnet_mixed_gran[3],
           marker='x', s=20**2, linewidth=5, color='red', alpha=.25)


ax.plot(resnet_mixed_sur[0], resnet_mixed_sur[1],
        label='Mixed Surrogate', ms=20, markeredgewidth=5, linewidth=5,
        color='blue')
ax.scatter(resnet_mixed_sur[2], resnet_mixed_sur[3],
           marker='x', s=20**2, linewidth=5, color='blue', alpha=.25)


ax.plot(resnet_mixed_sur_gran[0], resnet_mixed_sur_gran[1],
        label='Mixed Granular Surrogate', ms=20, markeredgewidth=5,
        linewidth=5, color='magenta')
ax.scatter(resnet_mixed_sur_gran[2], resnet_mixed_sur_gran[3],
           marker='x', s=20**2, linewidth=5, color='magenta', alpha=.25)


# plt.ylim(31, 38)
ax.set_xscale('log')
plt.xticks([1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4], [
    '1.2', '1.4', '1.6', '1.8', '2.0', '2.2', '2.4'])
ax.set_xlabel("Weight Size + Max Activation Size (MB)",
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
plt.savefig('figures/fitted_curve.png', dpi=300)
plt.close()
