from absl.testing import absltest
from absl.testing import parameterized
from absl import logging

import numpy as np
from collections import Counter
import torch
from scipy.stats import entropy

from utils import compute_amax_entropy

def compute_amax_entropy_torch(calib_hist, calib_bin_edges, num_bits, unsigned, stride=1, start_bin=128):
  # Original code from https://github.com/NVIDIA/TensorRT/blob/main/tools/\
  # pytorch-quantization/pytorch_quantization/calib/histogram.py 
  """Returns amax that minimizes KL-Divergence of the collected histogram"""

  # If calibrator hasn't collected any data, return none
  if calib_bin_edges is None and calib_hist is None:
    return None

  def _normalize_distr(distr):
    summ = np.sum(distr)
    if summ != 0:
      distr = distr / summ

  bins = calib_hist[:]
  bins[0] = bins[1]

  total_data = np.sum(bins)

  divergences = []
  arguments = []

  # we are quantizing to 128 values + sign if num_bits=8
  nbins = 1 << (num_bits - 1 + int(unsigned))

  starting = start_bin
  stop = len(bins)

  new_density_counts = np.zeros(nbins, dtype=np.float64)

  for i in range(starting, stop + 1, stride):
    new_density_counts.fill(0)
    space = np.linspace(0, i, num=nbins + 1)
    digitized_space = np.digitize(range(i), space) - 1

    digitized_space[bins[:i] == 0] = -1

    for idx, digitized in enumerate(digitized_space):
      if digitized != -1:
        new_density_counts[digitized] += bins[idx]

    counter = Counter(digitized_space)
    for key, val in counter.items():
      if key != -1:
        new_density_counts[key] = new_density_counts[key] / val

    new_density = np.zeros(i, dtype=np.float64)
    for idx, digitized in enumerate(digitized_space):
      if digitized != -1:
        new_density[idx] = new_density_counts[digitized]

    total_counts_new = np.sum(new_density) + np.sum(bins[i:])
    _normalize_distr(new_density)

    reference_density = np.array(bins[:len(digitized_space)])
    reference_density[-1] += np.sum(bins[i:])

    total_counts_old = np.sum(reference_density)
    if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
      raise RuntimeError("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(total_counts_new, total_counts_old, total_data))

    _normalize_distr(reference_density)

    ent = entropy(reference_density, new_density)
    divergences.append(ent)
    arguments.append(i)

  divergences = np.array(divergences)
  logging.debug("divergences={}".format(divergences))
  last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
  calib_amax = calib_bin_edges[last_argmin * stride + starting]
  calib_amax = torch.tensor(calib_amax.item()) #pylint: disable=not-callable

  return calib_amax


def entropy_test_data():
  return (
      dict(
          testcase_name = "small_data_bits8",
          dim_x = 512,
          dim_y = 512,
          bits = 8,
      ),
      # dict(
      #     testcase_name="medium_data_bits8",
      #     dimX=2048,
      #     dimY=2048,
      #     bits = 8,
      # ),
      # dict(
      #     testcase_name="large_data_bits8",
      #     dimX=10240,
      #     dimY=10240,
      #     bits = 8,
      # ),
    )



class EntropyInitTest(parameterized.TestCase):
  @parameterized.named_parameters(*entropy_test_data())
  def test_JaxEntropy_vs_Torch(self, dim_x, dim_y, bits):

    data = np.random.randn(dim_x, dim_y)

    hist, bin_edges = np.histogram(data, 2048)

    torch_amax = compute_amax_entropy_torch(hist, bin_edges, bits, True)

    jax_amax = compute_amax_entropy(hist, bin_edges, bits, True)

    self.assertEqual(torch_amax, jax_amax)


if __name__ == "__main__":
  absltest.main()
