from glob import glob

from tensorflow.core.util import event_pb2
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

pc_weight_noise = (
    "/afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/weight_noise_pc"
)
pc_act_noise = (
    "/afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/act_noise_pc"
)
pc_err_inpt_noise = (
    "/afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/err_inpt_noise_pc"
)
pc_err_weight_noise = (
    "/afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/err_weight_noise_pc"
)



bp_weight_noise = (
    "/afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/weight_noise_bp"
)
bp_act_noise = (
    "/afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/act_noise_bp"
)
bp_err_inpt_noise = (
    "/afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/err_inpt_noise_bp"
)
bp_err_weight_noise = (
    "/afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/err_weight_noise_bp"
)



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
    if len(glob(subdir + "/*tfevents*")) >= 1:
      if float(subdir.split("/")[-1].split('_')[0]) in data:

        data[float(subdir.split("/")[-1].split('_')[0])].append(read_tfevents(
            glob(subdir + "/*tfevents*")[0]
        ))
      else:
        data[float(subdir.split("/")[-1].split('_')[0])] = [read_tfevents(
            glob(subdir + "/*tfevents*")[0]
        )]

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
      sub_sample.append( np.sort(sub_val["eval_accuracy"])[-samples:] )

    x_obs.append(i)
    mean_obs.append(np.mean(sub_sample))
    std_obs.append(np.std(sub_sample))

  idx = np.argsort(x_obs)
  return np.array(mean_obs)[idx], np.array(std_obs)[idx], np.array(x_obs)[idx]


# samples = 3
# mean_pc_weight, std_obs_weight, x_pc_weight = mean_std_eval_acc(
#     pc_weight_noise, samples
# )
# mean_pc_act, std_obs_act, x_pc_act = mean_std_eval_acc(pc_act_noise, samples)
# mean_pc_err_inpt, std_obs_err_inpt, x_pc_err_inpt = mean_std_eval_acc(
#     pc_err_inpt_noise, samples
# )
# mean_pc_err_weight, std_obs_err_weight, x_pc_err_weight = mean_std_eval_acc(
#     pc_err_weight_noise, samples
# )

# # Plot PC

# #def plot_figure()
# fig, ax = plt.subplots(figsize=(8, 5.5))
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# ax.plot(x_pc_weight, mean_pc_weight, label="Weight")
# ax.fill_between(
#     x_pc_weight,
#     mean_pc_weight - std_obs_weight,
#     mean_pc_weight + std_obs_weight,
#     alpha=0.1,
# )

# ax.plot(x_pc_act, mean_pc_act, label="Activation")
# ax.fill_between(
#     x_pc_act, mean_pc_act - std_obs_act, mean_pc_act + std_obs_act, alpha=0.1
# )

# ax.plot(x_pc_err_inpt, mean_pc_err_inpt, label="Error Activation")
# ax.fill_between(
#     x_pc_err_inpt,
#     mean_pc_err_inpt - std_obs_err_inpt,
#     mean_pc_err_inpt + std_obs_err_inpt,
#     alpha=0.1,
# )

# ax.plot(x_pc_err_weight, mean_pc_err_weight, label="Error Weight")
# ax.fill_between(
#     x_pc_err_weight,
#     mean_pc_err_weight - std_obs_err_weight,
#     mean_pc_err_weight + std_obs_err_weight,
#     alpha=0.1,
# )

# plt.legend(
#     bbox_to_anchor=(0.5, 1.2), loc="upper center", ncol=2, frameon=False
# )
# plt.tight_layout()
# plt.savefig("figures/pc_noise_ablation.png")
# # plt.show()
# plt.close()


samples = 3
mean_pc_weight, std_obs_weight, x_pc_weight = mean_std_eval_acc(
    bp_weight_noise, samples
)
mean_pc_act, std_obs_act, x_pc_act = mean_std_eval_acc(bp_act_noise, samples)
mean_pc_err_inpt, std_obs_err_inpt, x_pc_err_inpt = mean_std_eval_acc(
    bp_err_inpt_noise, samples
)
mean_pc_err_weight, std_obs_err_weight, x_pc_err_weight = mean_std_eval_acc(
    bp_err_weight_noise, samples
)

# Plot PC

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.plot(x_pc_weight, mean_pc_weight, label="Weight")
ax.fill_between(
    x_pc_weight,
    mean_pc_weight - std_obs_weight,
    mean_pc_weight + std_obs_weight,
    alpha=0.1,
)

ax.plot(x_pc_act, mean_pc_act, label="Activation")
ax.fill_between(
    x_pc_act, mean_pc_act - std_obs_act, mean_pc_act + std_obs_act, alpha=0.1
)

ax.plot(x_pc_err_inpt, mean_pc_err_inpt, label="Error Activation")
ax.fill_between(
    x_pc_err_inpt,
    mean_pc_err_inpt - std_obs_err_inpt,
    mean_pc_err_inpt + std_obs_err_inpt,
    alpha=0.1,
)

ax.plot(x_pc_err_weight, mean_pc_err_weight, label="Error Weight")
ax.fill_between(
    x_pc_err_weight,
    mean_pc_err_weight - std_obs_err_weight,
    mean_pc_err_weight + std_obs_err_weight,
    alpha=0.1,
)

plt.legend(
    bbox_to_anchor=(0.5, 1.2), loc="upper center", ncol=2, frameon=False
)
plt.tight_layout()
plt.savefig("figures/bp_noise_ablation.png")
# plt.show()
plt.close()
