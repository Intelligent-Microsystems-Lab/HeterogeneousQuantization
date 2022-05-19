import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import tensorboard as tb
from packaging import version
import grpc
import string
from collections import OrderedDict

import numpy as np
import pandas as pd
import itertools

from absl import app

import pickle
import copy

import matplotlib.pyplot as plt

import jax
from jax import random
import jax.numpy as jnp

import ml_collections
from flax.core import unfreeze

from collections import OrderedDict

import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../mobilenetv2')

from train_utils import (  # noqa: E402
    create_model,
    create_train_state,
    restore_checkpoint,
)
import models as enet_models  # noqa: E402
import mobilenetv2.models as mbnet_model  # noqa: E402
import configs.vis_dummy as enet_conf  # noqa: E402
import configs.vis_dummy_mb as mb_conf  # noqa: E402


def load_data_mbnet():

  config = mb_conf.get_config()

  rng = random.PRNGKey(0)
  model_cls = getattr(mbnet_model, config.model)
  model = create_model(model_cls=model_cls,
                       num_classes=config.num_classes, config=config)

  rng, subkey = jax.random.split(rng, 2)
  state = create_train_state(
      subkey, config, model, config.image_size, lambda x: x)
  state = restore_checkpoint(state, '/Users/clemens/Desktop/mbnetv2_mixed_4_finetune_7')

  

  for i in range(16):
    print(i)
    x = state.params['quant_params']['InvertedResidual_'+str(i)]['QuantConv_0']['parametric_d_xmax_0']
    bits = np.ceil(np.log(x['dynamic_range'] / x['step_size']))
    print(np.unique(bits))

    try:
      x = state.params['quant_params']['InvertedResidual_'+str(i)]['QuantConv_1']['parametric_d_xmax_0']
      bits = np.ceil(np.log(x['dynamic_range'] / x['step_size']))
      print(np.unique(bits))
    except:
      pass

    try:
      x = state.params['quant_params']['InvertedResidual_'+str(i)]['QuantConv_2']['parametric_d_xmax_0']
      bits = np.ceil(np.log(x['dynamic_range'] / x['step_size']))
      print(np.unique(bits))
    except:
      pass

  # interesting 0depth, 1_0, 1depth, 3_0, 
  x = state.params['quant_params']['InvertedResidual_0']['QuantConv_0']['parametric_d_xmax_0'] # 32
  # x = state.params['quant_params']['MBConvBlock_1']['depthwise_conv2d']['parametric_d_xmax_0'] # 96 nope
  # x = state.params['quant_params']['MBConvBlock_1']['QuantConv_0']['parametric_d_xmax_0'] #96 nope
  # x = state.params['quant_params']['MBConvBlock_3']['QuantConv_0']['parametric_d_xmax_0'] # 144 nope

  bits = np.ceil(np.log(x['dynamic_range'] / x['step_size']))

  np.save('/Users/clemens/Desktop/channel_bits_mbnet.npy',(bits, x['dynamic_range']) )



def load_data_enet():

  config = enet_conf.get_config()

  rng = random.PRNGKey(0)
  model_cls = getattr(enet_models, config.model)
  model = create_model(model_cls=model_cls,
                       num_classes=config.num_classes, config=config)

  rng, subkey = jax.random.split(rng, 2)
  state = create_train_state(
      subkey, config, model, config.image_size, lambda x: x)
  state = restore_checkpoint(state, '/Users/clemens/Desktop/efficientnet-lite0_mixed_2.8_gran_sur_9')

  

  # for i in range(16):
  #   print(i)
  #   x = state.params['quant_params']['MBConvBlock_'+str(i)]['QuantConv_0']['parametric_d_xmax_0']
  #   bits = np.ceil(np.log(x['dynamic_range'] / x['step_size']))
  #   print(np.unique(bits))

  #   try:
  #     x = state.params['quant_params']['MBConvBlock_'+str(i)]['QuantConv_1']['parametric_d_xmax_0']
  #     bits = np.ceil(np.log(x['dynamic_range'] / x['step_size']))
  #     print(np.unique(bits))
  #   except:
  #     pass

  #   print('-')
  #   x = state.params['quant_params']['MBConvBlock_'+str(i)]['depthwise_conv2d']['parametric_d_xmax_0']
  #   bits = np.ceil(np.log(x['dynamic_range'] / x['step_size']))
  #   print(np.unique(bits))

  # interesting 0depth, 1_0, 1depth, 3_0, 
  x = state.params['quant_params']['MBConvBlock_0']['depthwise_conv2d']['parametric_d_xmax_0'] # 32
  # x = state.params['quant_params']['MBConvBlock_1']['depthwise_conv2d']['parametric_d_xmax_0'] # 96 nope
  # x = state.params['quant_params']['MBConvBlock_1']['QuantConv_0']['parametric_d_xmax_0'] #96 nope
  # x = state.params['quant_params']['MBConvBlock_3']['QuantConv_0']['parametric_d_xmax_0'] # 144 nope

  bits = np.ceil(np.log(x['dynamic_range'] / x['step_size']))

  np.save('/Users/clemens/Desktop/channel_bits.npy',(bits, x['dynamic_range']) )



def plot_additional_fig():
  font_size = 23
  gen_linewidth = 3
  plt.rc('font', family='Helvetica', weight='bold')

  #fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(
  #    14.4, 4.75), gridspec_kw={'width_ratios': [3,1]})

  fig4 = plt.figure( figsize=(14.4, 6.75)) #constrained_layout=True,
  gs = fig4.add_gridspec(ncols=5, nrows=4)
  
  ax_lr = fig4.add_subplot(gs[:1, :3])

  ax_acc = fig4.add_subplot(gs[1:, :3])

  ax_enet = fig4.add_subplot(gs[:2, 3:])

  ax_mbnet = fig4.add_subplot(gs[2:, 3:])



  for axis in ['top', 'bottom', 'left', 'right']:
    ax_lr.spines[axis].set_linewidth(gen_linewidth)

  for tick in ax_lr.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax_lr.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  
  #ax[0][0].set_xlabel("Train Step",
  #                 fontsize=font_size, fontweight='bold')

  ax_lr.spines["top"].set_visible(False)
  ax_lr.spines["right"].set_visible(False)

  ax_lr.xaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)
  ax_lr.yaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)

  # data_acc = pd.read_csv('figures/training_traces/run-train-tag-accuracy.csv', sep=',')
  # data_act = pd.read_csv('figures/training_traces/run-train-tag-act_size_sum.csv', sep=',')
  
  data_lr = pd.read_csv('figures/training_traces/pretrain-learning_rate.csv', sep=',')
  data_penalty = data_lr.copy()
  data_penalty['Value'] *= 0
  # data_penalty = pd.read_csv('figures/training_traces/pretrain-penalty_strength.csv', sep=',')

  # import pdb; pdb.set_trace()
  data_lr_t = pd.read_csv('figures/training_traces/run-train-tag-learning_rate.csv', sep=',')
  data_lr_t['Step'] += data_lr['Step'].iloc[-1]
  data_lr = pd.concat([data_lr, data_lr_t], ignore_index=True)

  data_penalty_t = pd.read_csv('figures/training_traces/run-train-tag-penalty_strength.csv', sep=',')
  data_penalty_t['Step'] += data_penalty['Step'].iloc[-1]
  data_penalty = pd.concat([data_penalty, data_penalty_t], ignore_index=True)


  data_lr_t = pd.read_csv('figures/training_traces/finetune-learning_rate.csv', sep=',')
  data_lr_t['Step'] += data_lr['Step'].iloc[-1]
  data_lr = pd.concat([data_lr, data_lr_t], ignore_index=True)

  # data_penalty = pd.read_csv('figures/training_traces/finetune-penalty_strength.csv', sep=',')
  data_penalty_t = data_lr_t.copy()
  data_penalty_t['Value'] *= 0
  data_penalty_t['Step'] += data_penalty['Step'].iloc[-1]
  data_penalty = pd.concat([data_penalty, data_penalty_t], ignore_index=True)

  data_penalty['Step'] = data_lr['Step']

  ax_lr.plot(data_lr['Step'], data_lr['Value'], label='Learning Rate', color = 'orange', linewidth=gen_linewidth,)
  
  ax_lr.set_yscale('log')
  
  ax_lr.axes.xaxis.set_visible(False)
  ax2 = ax_lr.twinx() 
  ax2.plot(data_penalty['Step'], data_penalty['Value'], label=r'Size Penalty ($\beta$)', color = 'blue', linewidth=gen_linewidth,)

  ax2.set_ylabel(r'$\beta$',
                   fontsize=font_size, fontweight='bold')


  ax_lr.set_ylabel("LR",
                   fontsize=font_size, fontweight='bold')


  ax2.plot([62464, 62464], [0, 1], color='grey', alpha=.75,
          linestyle='--', linewidth=3, zorder=0)
  ax2.plot([62464*2, 62464*2], [0, 1], color='grey', alpha=.75,
          linestyle='--', linewidth=3, zorder=0)

  ax2.set_yticks([0, 1])
  ax2.set_yticklabels(['0', '1e-4'])



  for axis in ['top', 'bottom', 'left', 'right']:
    ax_acc.spines[axis].set_linewidth(gen_linewidth)

  for tick in ax_acc.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax_acc.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')


  ax_acc.text(4_000, 62.5, 'Pre-Training', dict(size=23))
  ax_acc.text(67_500, 62, 'Quantization\n    Training', dict(size=23))
  ax_acc.text(130_000, 62.5, 'Finetuning', dict(size=23))
  
  ax_acc.set_xlabel("Train Step",
                   fontsize=font_size, fontweight='bold')
  #ax[0].set_xlabel("Train Step",
  #                 fontsize=font_size, fontweight='bold')

  ax_acc.set_ylabel("Accuracy (%)",
                   fontsize=font_size, fontweight='bold')



  # ax_acc.axes.xaxis.set_visible(False)
  ax_acc.set_xticks([0, 25_000, 50_000, 75_000, 100_000, 125_000, 150_000, 175_000])
  ax_acc.set_xticklabels(['0', '25k', '50k', '75k', '100k', '125k', '150k', '175k'])



  ax_acc.spines["top"].set_visible(False)
  ax_acc.spines["right"].set_visible(False)

  ax_acc.xaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)
  ax_acc.yaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)


  data_acc = pd.read_csv('figures/training_traces/pretrain-accuracy.csv', sep=',')

  data_acc_t = pd.read_csv('figures/training_traces/run-train-tag-accuracy.csv', sep=',')
  data_acc_t['Step'] += data_acc['Step'].iloc[-1]
  data_acc = pd.concat([data_acc, data_acc_t], ignore_index=True)

  data_acc_t = pd.read_csv('figures/training_traces/finetune-accuracy.csv', sep=',')
  data_acc_t['Step'] += data_acc['Step'].iloc[-1]
  data_acc = pd.concat([data_acc, data_acc_t], ignore_index=True)


  data_act = pd.read_csv('figures/training_traces/pretrain-act_size_sum.csv', sep=',')

  data_act_t = pd.read_csv('figures/training_traces/run-train-tag-act_size_sum.csv', sep=',')
  data_act_t['Step'] += data_act['Step'].iloc[-1]
  data_act = pd.concat([data_act, data_act_t], ignore_index=True)

  data_act_t = pd.read_csv('figures/training_traces/finetune-act_size_sum.csv', sep=',')
  data_act_t['Step'] += data_act['Step'].iloc[-1]
  data_act = pd.concat([data_act, data_act_t], ignore_index=True)


  data_wgt = pd.read_csv('figures/training_traces/pretrain-weight_size.csv', sep=',')

  data_wgt_t = pd.read_csv('figures/training_traces/run-train-tag-weight_size.csv', sep=',')
  data_wgt_t['Step'] += data_wgt['Step'].iloc[-1]
  data_wgt = pd.concat([data_wgt, data_wgt_t], ignore_index=True)

  data_wgt_t = pd.read_csv('figures/training_traces/finetune-weight_size.csv', sep=',')
  data_wgt_t['Step'] += data_wgt['Step'].iloc[-1]
  data_wgt = pd.concat([data_wgt, data_wgt_t], ignore_index=True)



  ax_acc.plot(data_acc['Step'], data_acc['Value']*100, label='Accuracy', color = 'red', linewidth=gen_linewidth,)
  
  ax_acc.plot([62464, 62464], [np.min(data_acc['Value']*100),np.max(data_acc['Value']*100)], color='grey', alpha=.75,
          linestyle='--', linewidth=3, zorder=0)
  ax_acc.plot([62464*2, 62464*2], [np.min(data_acc['Value']*100),np.max(data_acc['Value']*100)], color='grey', alpha=.75,
          linestyle='--', linewidth=3, zorder=0)


  ax3 = ax_acc.twinx() 
  ax3.plot(data_act['Step'], (data_act['Value'] + data_wgt['Value'])/1000, label='Memory Footprint', color = 'green', linewidth=gen_linewidth,)

  ax3.set_ylabel("MB",
                   fontsize=font_size, fontweight='bold')


  bits, xmax = np.load('/Users/clemens/Desktop/channel_bits.npy', allow_pickle=True)


  ax_enet.spines["top"].set_visible(False)
  ax_enet.spines["right"].set_visible(False)
  ax_enet.spines["bottom"].set_visible(False)
  ax_enet.spines["left"].set_visible(False)

  # ax_enet.set_xlabel("# Channel",
  #                  fontsize=font_size, fontweight='bold')

  ax_enet.xaxis.set_tick_params(width=3, length=10, labelsize=font_size)
  ax_enet.yaxis.set_tick_params(width=3, length=10, labelsize=font_size)

  ax_enet.spines['left'].set_position('zero')

  for axis in ['top', 'bottom', 'left', 'right']:
    ax_enet.spines[axis].set_linewidth(5)

  for tick in ax_enet.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax_enet.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  localw_max_bits = 4
  ax_enet.plot([0, 32 + 1], [2/4, 2/4], color='grey', alpha=.5,
          linestyle='--', linewidth=3, zorder=0)
  ax_enet.plot([0, 32 + 1], [4/4, 4/4], color='grey', alpha=.5,
          linestyle='--', linewidth=3, zorder=0)


  ax_enet.plot([0, 32 + 1], [-2/5, -2/5], color='grey', alpha=.5,
          linestyle='--', linewidth=3, zorder=0)
  ax_enet.plot([0, 32 + 1], [-4/5, -4/5], color='grey', alpha=.5,
          linestyle='--', linewidth=3, zorder=0)

  for i in range(bits.shape[0]):
    ax_enet.bar(i+.5, (bits[i]+1)/4, width=1.0, color='m', edgecolor='black', zorder=10)

  # import pdb; pdb.set_trace()
  for i in range(bits.shape[0]):
    ax_enet.bar(i+.5, -xmax[i]/xmax.max(), width=1.0, color='c', edgecolor='black', zorder=10)


  ax_enet.text(-8, 0.3, '# Bits', dict(size=23), rotation=90)
  ax_enet.text(-8, -.7, 'Xmax', dict(size=23), rotation=90)
  ax_enet.annotate('', xy=(0, 0), xytext=(32 + 2, 0), arrowprops=dict(
      arrowstyle='<-, head_width=.3, head_length=1.', lw=3), zorder=20)
  ax_enet.annotate('', xy=(0, -1.10), xytext=(0, 1.25), arrowprops=dict(
      arrowstyle='<->, head_width=.3,  head_length=1.', lw=3), zorder=20)

  #ax[2].set_ylim(-.97, 1.15)
  ax_enet.set_yticks([1., .5, 0., -2/5, -4/5])
  ax_enet.set_yticklabels([4, 2, 0, '2.3', '4.7',])
  # ax[2].text(37, -.82, name, dict(size=23))
  ax_enet.text(8, -.75, 'EfficientNet-Lite0', dict(size=23))
  ax_enet.axes.xaxis.set_visible(False)

  # ax2.set_yticks([1., .5, 0])
  # ax2.set_yticklabels(['1.0', '0.5','0.0'])





  bits, xmax = np.load('/Users/clemens/Desktop/channel_bits_mbnet.npy', allow_pickle=True)

  ax_mbnet.spines["top"].set_visible(False)
  ax_mbnet.spines["right"].set_visible(False)
  ax_mbnet.spines["bottom"].set_visible(False)
  ax_mbnet.spines["left"].set_visible(False)

  # ax_mbnet.set_xlabel("# Channel",
  #                  fontsize=font_size, fontweight='bold')

  ax_mbnet.xaxis.set_tick_params(width=3, length=10, labelsize=font_size)
  ax_mbnet.yaxis.set_tick_params(width=3, length=10, labelsize=font_size)

  ax_mbnet.spines['left'].set_position('zero')

  for axis in ['top', 'bottom', 'left', 'right']:
    ax_mbnet.spines[axis].set_linewidth(5)

  for tick in ax_mbnet.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax_mbnet.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  localw_max_bits = 5
  ax_mbnet.plot([0, 32 + 1], [2/5, 2/5], color='grey', alpha=.5,
          linestyle='--', linewidth=3, zorder=0)
  ax_mbnet.plot([0, 32 + 1], [4/5, 4/5], color='grey', alpha=.5,
          linestyle='--', linewidth=3, zorder=0)


  ax_mbnet.plot([0, 32 + 1], [-2/5, -2/5], color='grey', alpha=.5,
          linestyle='--', linewidth=3, zorder=0)
  ax_mbnet.plot([0, 32 + 1], [-4/5, -4/5], color='grey', alpha=.5,
          linestyle='--', linewidth=3, zorder=0)

  for i in range(bits.shape[0]):
    ax_mbnet.bar(i+.5, (bits[i]+1)/5, width=1.0, color = 'm', edgecolor='black', zorder=10)

  # import pdb; pdb.set_trace()
  for i in range(bits.shape[0]):
    ax_mbnet.bar(i+.5, -xmax[i]/xmax.max(), width=1.0, color='c', edgecolor='black', zorder=10)


  ax_mbnet.text(-8, 0.3, '# Bits', dict(size=23), rotation=90)
  ax_mbnet.text(-8, -.7, 'Xmax', dict(size=23), rotation=90)
  ax_mbnet.annotate('', xy=(0, 0), xytext=(32 + 2, 0), arrowprops=dict(
      arrowstyle='<-, head_width=.3, head_length=1.', lw=3), zorder=20)
  ax_mbnet.annotate('', xy=(0, -1.1), xytext=(0, 1.2), arrowprops=dict(
      arrowstyle='<->, head_width=.3,  head_length=1.', lw=3), zorder=20)

  #ax[2].set_ylim(-.97, 1.15)
  ax_mbnet.set_yticks([4/5, 2/5, 0., -2/5, -4/5])
  ax_mbnet.set_yticklabels([4, 2, 0, '1.5', '3.0',])
  # ax[2].text(37, -.82, name, dict(size=23))

  ax_mbnet.text(13, -.74, 'MobileNetV2', dict(size=23))
  ax_mbnet.set_xticks([])
  ax_mbnet.set_xticklabels([])
  ax_mbnet.set_xlabel("# Channel",
                   fontsize=font_size, fontweight='bold')








  for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(gen_linewidth)
    ax3.spines[axis].set_linewidth(gen_linewidth)

  for tick in ax2.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax2.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(gen_linewidth)


  ax2.xaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)
  ax2.yaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)

  ax3.xaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)
  ax3.yaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)


  for tick in ax3.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax3.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  for axis in ['top', 'bottom', 'left', 'right']:
    ax3.spines[axis].set_linewidth(gen_linewidth)


  ax2.spines["top"].set_visible(False)
  ax3.spines["top"].set_visible(False)

  handles, labels = ax_acc.get_legend_handles_labels()


  handles_t, labels_t = ax2.get_legend_handles_labels()
  handles += handles_t

  handles_t, labels_t = ax_lr.get_legend_handles_labels()
  handles += handles_t

  handles_t, labels_t = ax3.get_legend_handles_labels()
  handles += handles_t


  fig4.legend(
      handles=handles,
      bbox_to_anchor=(.0, 1.02, 1., .05),
      loc="upper center",
      ncol=4,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size}
  )


  plt.tight_layout()
  plt.savefig('figures/additional_figure.png', dpi=300, bbox_inches='tight')
  plt.close()



if __name__ == '__main__':
  major_ver, minor_ver, _ = version.parse(tb.__version__).release
  assert major_ver >= 2 and minor_ver >= 3, \
      "This notebook requires TensorBoard 2.3 or later."
  print("TensorBoard version: ", tb.__version__)

  mpl.rcParams['font.family'] = 'sans-serif'
  mpl.rcParams['font.sans-serif'] = 'Helvetica'
  mpl.rcParams['font.weight'] = 'bold'
  mpl.rcParams['mathtext.fontset'] = 'custom'
  mpl.rcParams['mathtext.rm'] = 'sans'
  mpl.rcParams['mathtext.it'] = 'sans:bold'
  mpl.rcParams['mathtext.default'] = 'bf'

  # load_data_mbnet()
  # load_data_enet()
  plot_additional_fig()
