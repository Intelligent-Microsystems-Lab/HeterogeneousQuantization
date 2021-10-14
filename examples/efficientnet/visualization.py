import matplotlib.pyplot as plt
import tensorboard as tb
from packaging import version


# Validation Runs.

enets_fp = {
    'label': 'EfficientNet Scaling (FP32)',
    'color': 'b',
    0: 'BRj9fv5PR0yAWkD4z0p5FQ',
    1: 'QRMPo8cVQRqk01JbKZOMjw',
    2: 'DZXKGFneSoW8rj5qZZz3LQ',
    3: 'dD3zay4XTYm6ltpNTGocDg',
    4: '4VwTvygFQ2WFG74GlF8Tqw',
}

enets_int8 = {
    'label': 'EfficientNet Scaling (INT8)',
    'color': 'g',
    0: 'L2wx6i0dRly0LTG3sA9qpg',
    1: 'oXPvlPrSQkKyZlivUrby7w',
    2: 'KMC8ULhbQviDC5LN1aj4dA',
    3: '7hgKgc31QXm7rVOX8hm0rg',
    4: 'G6PJWXMRQyiAMZCEyVfVsA',
}


# Parametric Quant

enet0_lsq = {
    'label': 'LSQ',
    'color': 'g',
    2: 'mYPAx8Z5QXmyeTx3prbdfQ',
    3: 's8rGb5dCQBWPnD1QyCVZsQ',
    4: 'wY16VBQXSpOcj7mClBMjfA',
    5: 'SGwFwwYaRqSXov6vgGwV9Q',
    6: 'xCru2mpURnOqzAsYRWvrdA',
    7: 'dEG44pIzRQmAA83QnCv52w',
    8: 'gS1RSoyBRfOGm4pGuBJ8BQ',
}


# Non-Parametric Quant

enet0_dynamic_init_max = {
    'label': 'Dynamic Init Max',
    'color': 'b',
    # 2:'IvvJGLjvS7SPDJsLC0jFLA',
    # 3:'pvKoZRvnSG6qfEszYE9JpQ',
    # 4:'rja3bfROSNCAR3rwM2QJBQ',
    # 5:'gwa1KzUVRLKb2ux6hQXHjQ',
    # 6:'RG9CyhLTQQWlpvGO0AqsLg',
    # 7:'YkYjgmu3TVajPWNvRohllg',
    # 8:'80CzYZCOQkOHgFNoRqFAkw',

    2: 'OmSmfq59S0aQYMmD7SYFjg',
    3: 'YKZxhW5ETg6M7F6EDO06bg',
    4: 'waHr3rS9R52JeWCbz1Lntw',
    5: '3fXh0YayQLuZzfjAWKN1kQ',
    6: 'wvkqUkAmTNunCBCdLaq6qw',
    7: 'agYwfLAgRXCa692n5t5R4A',
    8: 'rijJczuQQrWpxiz7EhVMdQ',
}


enet0_dynamic_init_double_mean = {
    'label': 'Dynamic Init Double Mean',
    'color': 'g',
    2: 'HA3kLdFAQReV8IRFwOLnig',
    3: 'aVJ26QKSRLqujYJ2uwlnBQ',
    4: 'FAfkPLlXRLKpfX72gqFWSg',
    5: 'ADvNNAYvSpmyXqsvv3g3hg',
    6: 'Ul3KxPh5TXSamryS5woj0A',
    7: 'nZVltBcqRPuXKJSeKDt0Ow',
    8: 'ZmwIrSA2SvaLhv85BtJwEw',
}

enet0_dynamic_init_gaussian = {
    'label': 'Dynamic Init Gaussian',
    'color': 'r',
    2: 'qS66vBsZQIqXnSjiGabI7g',
    3: 'YtVzqbnzSKmmdKz0pHof2Q',
    4: 'PCogVoW9TwO6WFGeVX31kg',
    5: 'ePpTK8YzQPG6kNFPq8MYrQ',
    6: 'V9yGsRBXQ0ePDXhqjzMJXw',
    7: 'dpHy0z8qS6aL5XED2CdEmQ',
    8: 've9KwJfYRxOiJ5elyNZyOQ',
}


enet0_static_init_max = {
    'label': 'Static Init Max',
    'color': 'c',
    2: '6Wd3QTwfRrahD5wHGV0yCw',
    3: 'goSVltnRRQyJU8QHyJnKYw',
    4: 'nLZUk2fpQ3qgzBGCJgksOg',
    5: '36lJJc0LRCayqMTAXROPiw',
    6: 'sjrggkHXSAGqBFzzsjHaPw',
    7: 'bxM6szNMSra2X4XkLkewNg',
    8: 'YTVldyMXR32Rj9eBXgqaXw',
}


enet0_static_init_double_mean = {
    'label': 'Static Init Double Mean',
    'color': 'm',
    2: 'kJwP08xMQsK1wrQJrxUE9A',
    3: 'cYvs4lQzR9mEyM8sP3szGQ',
    4: 'kJpmA5fyTMCeK6kTkpJoAw',
    5: 'NpC0BZnpRtyfoJnCOW6gdQ',
    6: '1e8dB7GpR62qD4gPat0mfg',
    7: 'uyRv8qxqSWeOI9GlkiW2Zw',
    8: 'PGYSZAO6TVKDaaMkRWjs3A',
}

enet0_static_init_gaussian = {
    'label': 'Static Init Gaussian',
    'color': 'k',
    2: 'm3uqrpJvQfOeZ1Vo8DM9SQ',
    3: 'mtjXBQOeQha1f2Al842B1w',
    4: 'NujiWt8nS6SdePJHK29jQw',
    5: 'gk2ERBL0ReKFaqDIkIMB6Q',
    6: 'c0co5awbRaWIZV9zc4f1Ag',
    7: 'yFLNoPx2Q4yhqnrpWQaY0A',
    8: '0tlDzZrqQC6KNXADNYw4rQ',
}


enet0_static_init_max_surrogate = {
    'label': 'Static Init Max Surrogate',
    'color': 'c',
    'linestyle': 'dashed',
    2: '',
    3: '',
    4: '',
    5: '',
    6: '',
    7: '',
    8: '',
}

enet0_dynamic_init_max_surrogate = {
    'label': 'Dynamic Init Max Surrogate',
    'color': 'b',
    'linestyle': 'dashed',
    2: '6VznjGRiQvegtlVIqYG1jA',
    3: 'YeQa1zJwQ7KavXL4nGaCpA',
    4: 'FnP7qU9BQta2OdRlT68M2A',
    5: '3hPRiW3jRo6iLKTjoaqTQQ',
    6: 'JNqKiV5WT92eOjrE2GE9Mg',
    7: 'SpatG4eBRdS2op9rdnN9vA',
    8: 's3fQeZZ8RZ6btvGhyRjjPg',
}


def get_best_eval(experiment_id):
  experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
  df = experiment.get_scalars()

  data = df[df['run'] == 'eval']
  return data[data['tag'] == 'accuracy']['value'].max()


def plot_line(ax, res_dict):
  linestyle = 'solid'
  x = []
  y = []
  for key, value in res_dict.items():
    if key == 'label':
      label = value
    elif key == 'color':
      color = value
    elif key == 'linestyle':
      linestyle = value
    else:
      y.append(get_best_eval(value))
      x.append(key)

  ax.plot(x, y, color=color, marker='x', label=label,
          linestyle=linestyle, ms=20, markeredgewidth=5, linewidth=5)


def plot_bits_vs_acc(list_of_dicts, name):
  font_size = 26

  fig, ax = plt.subplots(figsize=(13, 9.8))
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

  for v in list_of_dicts:
    plot_line(ax, v)

  ax.set_xlabel("Bits", fontsize=font_size, fontweight='bold')
  ax.set_ylabel("Eval Accuracy (%)", fontsize=font_size, fontweight='bold')
  plt.legend(
      bbox_to_anchor=(0.5, 1.2),
      loc="upper center",
      ncol=2,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size}
  )
  plt.tight_layout()
  plt.savefig(name)
  plt.close()


if __name__ == '__main__':
  major_ver, minor_ver, _ = version.parse(tb.__version__).release
  assert major_ver >= 2 and minor_ver >= 3, \
      "This notebook requires TensorBoard 2.3 or later."
  print("TensorBoard version: ", tb.__version__)

  plot_bits_vs_acc([enet0_dynamic_init_max, enet0_dynamic_init_double_mean,
                    enet0_dynamic_init_gaussian], 'figures/dynamic_init.png')

  plot_bits_vs_acc([enet0_static_init_max, enet0_static_init_double_mean,
                    enet0_static_init_gaussian], 'figures/static_init.png')

  plot_bits_vs_acc([enet0_dynamic_init_max_surrogate, enet0_static_init_max,
                    enet0_dynamic_init_max], 'figures/surrogate.png')
