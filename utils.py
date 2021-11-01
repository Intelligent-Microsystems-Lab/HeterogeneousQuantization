# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any
from collections import Counter

import time

def sse_loss(x, y):
  return jnp.sum((x - y) ** 2)


def kl(p: Any, q: Any, eps: float = 2 ** -17) -> Any:
  # Copied from https://objax.readthedocs.io/en/latest/_modules/objax/\
  # functional/divergence.html
  """Calculates the Kullback-Leibler divergence between arrays p and q."""
  return p.dot(jnp.log(p + eps) - jnp.log(q + eps))


def compute_amax_entropy(calib_hist, calib_bin_edges, num_bits, unsigned,
  stride=1, start_bin=128):
  # Copied and modified from https://github.com/NVIDIA/TensorRT/blob/main/\
  # tools/pytorch-quantization/pytorch_quantization/calib/histogram.py
  """Returns amax that minimizes KL-Divergence of the collected histogram"""

  # If calibrator hasn't collected any data, return none
  if calib_bin_edges is None and calib_hist is None:
      return None

  def _normalize_distr(distr):
    summ = jnp.sum(distr)
    return jnp.where(summ == 0, distr, distr / summ)
    # summ = np.sum(distr)
    # if summ != 0:
    #   distr = distr / summ

  bins = calib_hist#[:]
  #bins[0] = bins[1]

  total_data = np.sum(bins)

  divergences = []
  arguments = []

  # we are quantizing to 128 values + sign if num_bits=8
  nbins = 1 << (num_bits - 1 + int(unsigned))

  starting = start_bin
  stop = len(bins)

  #new_density_counts = jnp.zeros(nbins, dtype=np.float64)

  for i in range(starting, stop + 1, stride):
    # new_density_counts.fill(0)
    new_density_counts = jnp.zeros(nbins)
    space = np.linspace(0, i, num=nbins + 1)
    digitized_space = np.digitize(range(i), space) - 1

    digitized_space = jnp.where(bins[:i] == 0, -1, digitized_space)
    #digitized_space[bins[:i] == 0] = -1

    
    for idx, digitized in enumerate(digitized_space):
      new_density_counts = new_density_counts.at[digitized].set(jnp.where(digitized != -1, bins[idx], 0.) )
      #if digitized != -1:
      #  new_density_counts[digitized] += bins[idx]
    
    #counter = Counter(digitized_space)
    

    #
    # Costly
    #
    start_time = time.time()
    uval, ucount = jnp.unique(digitized_space, return_counts=True)

    #import pdb; pdb.set_trace()
    #new_density_counts = 

    for key, val in zip(uval, ucount):
      new_density_counts = new_density_counts.at[key].divide(jnp.where(key!=-1, val, 1.)) #(jnp.where(key != -1, new_density_counts[key]/val,new_density_counts[key]  ))
    #print("2--- %s seconds ---" % (time.time() - start_time))
    # for key, val in counter.items():
    #   if key != -1:
    #     new_density_counts[key] = new_density_counts[key] / val

    #
    # Costly
    #
    start_time = time.time()
    new_density = jnp.zeros(i)
    for idx, digitized in enumerate(digitized_space):
      new_density = new_density.at[idx].set(jnp.where(digitized != -1, new_density_counts[digitized], 0.))
    #print("3--- %s seconds ---" % (time.time() - start_time))
      #if digitized != -1:
      #  new_density[idx] = new_density_counts[digitized]

    total_counts_new = np.sum(new_density) + np.sum(bins[i:])
    new_density = _normalize_distr(new_density)

    reference_density = jax.lax.slice(bins, [0], [len(digitized_space)])
    # reference_density = np.array(bins[:len(digitized_space)])
    reference_density = reference_density.at[-1].set(reference_density[-1] + np.sum(bins[i:]))
    # reference_density[-1] += np.sum(bins[i:])

    # total_counts_old = np.sum(reference_density)
    # if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
    #   raise RuntimeError("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(
    #         total_counts_new, total_counts_old, total_data))

    reference_density = _normalize_distr(reference_density)

    ent = kl(reference_density, new_density)
    divergences.append(ent)
    arguments.append(i)

  divergences = np.array(divergences)
  logging.debug("divergences={}".format(divergences))
  last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
  calib_amax = calib_bin_edges[last_argmin * stride + starting]
  #calib_amax = torch.tensor(calib_amax.item()) #pylint: disable=not-callable

  return calib_amax
