from glob import glob
import argparse

from tensorflow.core.util import event_pb2
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--basedir", type=str, default="/afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps_t1/",
                    help="Base dir with sweep results.")
parser.add_argument("--samples", type=int, default=1,
                    help="Number of samples from each trial.")
args = parser.parse_args()


def read_tfevents(path):
  data = {}
  for batch in tf.data.TFRecordDataset(path):
    event = event_pb2.Event.FromString(batch.numpy())
    for value in event.summary.value:
      if value.tag not in data:
        data[value.tag] = []
        # data[value.tag+'_step'] = []
      data[value.tag].append((tf.make_ndarray(value.tensor).tolist()))
      # data[value.tag+'_step'].append(event.step)
  return data


def read_data_from_dir(path):
  data = {}

  for subdir in glob(path + "/*"):
    if len(glob(subdir + "/eval/*tfevents*")) >= 1:
      if float(subdir.split("/")[-1].split("_")[0]) in data:
        data[float(subdir.split("/")[-1].split("_")[0])].append(
            read_tfevents(glob(subdir + "/eval/*tfevents*")[0])
        )
      else:
        data[float(subdir.split("/")[-1].split("_")[0])] = [
            read_tfevents(glob(subdir + "/eval/*tfevents*")[0])
        ]

  return data


# extract eval acc mean std
def mean_std_eval_acc(path, samples):
  data = read_data_from_dir(path)

  mean_obs = []
  std_obs = []
  x_obs = []
  for i, val in data.items():
    if val == {}:
      continue

    sub_sample = []
    for sub_val in val:
      sub_sample.append(np.sort(sub_val["accuracy"])[-samples:])

    x_obs.append(i)
    mean_obs.append(np.mean(sub_sample))
    std_obs.append(np.std(sub_sample))

  idx = np.argsort(x_obs)
  return (
      np.array(mean_obs)[idx],
      np.array(std_obs)[idx],
      np.array(x_obs)[idx],
  )


def plot_curves(curve_fnames, title_str, png_fname, xaxis, samples):
  fig, ax = plt.subplots(figsize=(8, 5.5))
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  for key, value in curve_fnames.items():
    mean, std, x = mean_std_eval_acc(value, samples)

    ax.plot(x, mean, label=key)
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.1,
    )

  ax.set_xlabel(xaxis)
  ax.set_ylabel("Eval Acc")
  plt.legend(
      bbox_to_anchor=(0.5, 1.2), loc="upper center", ncol=2, frameon=False
  )
  plt.tight_layout()
  plt.savefig("figures/"+png_fname+".png")
  plt.close()

def plot_diff(curve1, curve2, title_str, png_fname, xaxis, samples):
  fig, ax = plt.subplots(figsize=(8, 5.5))
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  for key in curve1.keys() & curve2.keys():
    mean1, std1, x1 = mean_std_eval_acc(curve1[key], samples)
    mean2, std2, x2 = mean_std_eval_acc(curve2[key], samples)

    try:
      assert (x1 == x2).all()
    except:
      continue

    ax.plot(x1, mean1-mean2, label=key)
    import pdb; pdb.set_trace()
    #ax.fill_between(
    #    x,
    #    mean - std,
    #    mean + std,
    #    alpha=0.1,
    #)

  ax.set_xlabel(xaxis)
  ax.set_ylabel("Eval Acc")
  plt.legend(
      bbox_to_anchor=(0.5, 1.2), loc="upper center", ncol=2, frameon=False
  )
  plt.tight_layout()
  plt.savefig("figures/"+png_fname+".png")
  plt.close()

if __name__ == "__main__":
  pc_noise = {
      'Weights FWD': args.basedir+"/weight_noise_pc",
      'Activations FWD': args.basedir+"/act_noise_pc",
      'Error Activations': args.basedir+"/err_inpt_noise_pc",
      'Error Weights': args.basedir+"/err_weight_noise_pc",
      'Weights BWD': args.basedir+"/weight_bwd_noise_pc",
      'Activations BWD': args.basedir+"/act_bwd_noise_pc",
      'Value Node': args.basedir+"/val_noise_pc",
  }

  pc_bits = {
      'Weights FWD': args.basedir+"/weight_bits_pc",
      'Activations FWD': args.basedir+"/act_bits_pc",
      'Error Activations': args.basedir+"/err_inpt_bits_pc",
      'Error Weights': args.basedir+"/err_weight_bits_pc",
      'Weights BWD': args.basedir+"/weight_bwd_bits_pc",
      'Activations BWD': args.basedir+"/act_bwd_bits_pc",
      'Value Node': args.basedir+"/val_bits_pc",
  }

  bp_noise = {
      'Weights FWD': args.basedir+"/weight_noise_bp",
      'Activations FWD': args.basedir+"/act_noise_bp",
      'Error Activations': args.basedir+"/err_inpt_noise_bp",
      'Error Weights': args.basedir+"/err_weight_noise_bp",
      'Weights BWD': args.basedir+"/weight_bwd_noise_bp",
      'Activations BWD': args.basedir+"/act_bwd_noise_bp",
      'Value Node': args.basedir+"/val_noise_bp",
  }

  bp_bits = {
      'Weights FWD': args.basedir+"/weight_bits_bp",
      'Activations FWD': args.basedir+"/act_bits_bp",
      'Error Activations': args.basedir+"/err_inpt_bits_bp",
      'Error Weights': args.basedir+"/err_weight_bits_bp",
      'Weights BWD': args.basedir+"/weight_bwd_bits_bp",
      'Activations BWD': args.basedir+"/act_bwd_bits_bp",
      'Value Node': args.basedir+"/val_bits_bp",
  }

  #plot_curves(pc_noise, "Predictive Coding Noise", 'pc_noise', 'Noise', args.samples)
  #plot_curves(pc_bits, "Predictive Coding Quantization", 'pc_bits', 'Bits', args.samples)
  #plot_curves(bp_noise, "Backpropagation Noise", 'bp_noise', 'Noise', args.samples)
  #plot_curves(bp_bits, "Backpropagation Quantization", 'bp_bits', 'Bits', args.samples)

  plot_diff(bp_noise, pc_noise, "Differences Noise", 'diff_noise', 'Noise', args.samples)
  plot_diff(bp_bits, pc_bits, "Differences Quantization", 'diff_bits', 'Bits', args.samples)
