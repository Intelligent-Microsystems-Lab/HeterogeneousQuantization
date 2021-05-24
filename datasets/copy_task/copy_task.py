"""copy_task dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import jax
import jax.numpy as jnp
import numpy as np

# TODO(copy_task): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(copy_task): BibTeX citation
_CITATION = """
@article{graves2016hybrid,
  title={Hybrid computing using a neural network with dynamic external memory},
  author={Graves, Alex and Wayne, Greg and Reynolds, Malcolm and Harley, Tim
   and Danihelka, Ivo and Grabska-Barwi{\'n}ska, Agnieszka and Colmenarejo,
    Sergio G{\'o}mez and Grefenstette, Edward and Ramalho, Tiago and Agapiou,
     John and others},
  journal={Nature},
  volume={538},
  number={7626},
  pages={471--476},
  year={2016},
  publisher={Nature Publishing Group}
}
"""


def bitstring_readable(data, batch_size, model_output=None, whole_batch=False):
    """Produce a human readable representation of the sequences in data.
    Args:
      data: data to be visualised
      batch_size: size of batch
      model_output: optional model output tensor to visualize alongside data.
      whole_batch: whether to visualise the whole batch. Only the first sample
          will be visualized if False
    Returns:
      A string used to visualise the data batch
    """

    def _readable(datum):
        return (
            "+" + " ".join(["-" if x == 0 else "%d" % x for x in datum]) + "+"
        )

    obs_batch = data["observations"]
    targ_batch = data["target"]

    iterate_over = range(batch_size) if whole_batch else range(1)
    batch_strings = []
    for batch_index in iterate_over:
        obs = obs_batch[batch_index, :, :]
        targ = targ_batch[batch_index, :, :]

        obs_channels = range(obs.shape[1])
        targ_channels = range(targ.shape[1])
        obs_channel_strings = [_readable(obs[:, i]) for i in obs_channels]
        targ_channel_strings = [_readable(targ[:, i]) for i in targ_channels]

        readable_obs = "Observations:\n" + "\n".join(obs_channel_strings)
        readable_targ = "Targets:\n" + "\n".join(targ_channel_strings)
        strings = [readable_obs, readable_targ]

        if model_output is not None:
            output = model_output[batch_index, :, :]
            output_strings = [_readable(output[:, i]) for i in targ_channels]
            strings.append("Model Output:\n" + "\n".join(output_strings))

        batch_strings.append("\n\n".join(strings))

    return "\n" + "\n\n\n\n".join(batch_strings)


def to_human_readable(data, model_output=None, whole_batch=False):
    obs = data["observations"]
    batch_size, _, _ = data["observations"].shape
    obs = np.concatenate([obs[:, :, :-1], obs[:, :, -1:]], axis=2)
    data["observations"] = obs
    return bitstring_readable(data, batch_size, model_output, whole_batch)


class CopyTask(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for copy_task dataset."""

    RELEASE_NOTES = {"1.20.6": "Initial release."}
    NUM_BITS: int = 6
    MIN_LENGTH: int = 4
    MAX_LENGTH: int = 4
    MIN_REPEATS: int = 1
    MAX_REPEATS: int = 1
    DS_SIZE: int = 1024
    # SEED: int = 42
    rng = jax.random.PRNGKey(42)
    VERSION = tfds.core.Version("1." + str(NUM_BITS) + "." + str(MAX_LENGTH))

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        max_length = (self.MAX_LENGTH) * 2 + 3

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images ...
                    "observations": tfds.features.Tensor(
                        shape=(max_length, self.NUM_BITS + 2), dtype=tf.bool
                    ),
                    "target": tfds.features.Tensor(
                        shape=(max_length, self.NUM_BITS + 1), dtype=tf.bool
                    ),
                    "mask": tfds.features.Tensor(
                        shape=(max_length,), dtype=tf.bool
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=(
                "observations",
                "target",
            ),  # Set to `None` to disable
            homepage="https://github.com/deepmind/dnc/blob/master/dnc/"
            "repeat_copy.py",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        return {
            "train": self._generate_examples(60000),
            "test": self._generate_examples(10000),
        }

    def _generate_examples(self, split_size):
        # short-hand for private fields.
        min_length, max_length = self.MIN_LENGTH, self.MAX_LENGTH
        num_bits = self.NUM_BITS

        # We reserve one dimension for the num-repeats and one for the
        # start-marker.
        full_obs_size = num_bits + 2
        # We reserve one target dimension for the end-marker.
        full_targ_size = num_bits + 1
        start_end_flag_idx = full_obs_size - 2
        num_repeats_channel_idx = full_obs_size - 1

        # Samples each batch index's sequence length and the number of repeats.
        self.rng, input_rng = jax.random.split(self.rng)
        sub_seq_length_batch = jnp.array(
            jax.random.uniform(
                input_rng, [self.DS_SIZE], minval=min_length, maxval=max_length
            ),
            dtype=jnp.int32,
        )
        # Pads all the batches to have the same total sequence length.
        total_length_batch = sub_seq_length_batch * 2 + 3

        max_length_batch = int((self.MAX_LENGTH) * 2 + 3)
        residual_length_batch = max_length_batch - total_length_batch

        for sample_idx in range(split_size):

            obs_tensors = []
            targ_tensors = []
            mask_tensors = []

            # Generates patterns for each batch element independently.
            # for batch_index in range(batch_size):
            sub_seq_len = sub_seq_length_batch[sample_idx]
            num_reps = 1  # num_repeats_batch[sample_idx]

            # The observation pattern is a sequence of random binary vectors.
            obs_pattern_shape = [sub_seq_len, num_bits]
            self.rng, input_rng = jax.random.split(self.rng)
            obs_pattern = jnp.array(
                jnp.array(
                    jax.random.uniform(
                        input_rng, obs_pattern_shape, minval=0, maxval=2
                    ),
                    dtype=jnp.int32,
                ),
                dtype=jnp.float32,
            )

            # The target pattern is the observation pattern repeated n times.
            # Some reshaping is required to accomplish the tiling.
            targ_pattern_shape = [sub_seq_len * num_reps, num_bits]
            flat_obs_pattern = jnp.reshape(obs_pattern, [-1])
            flat_targ_pattern = jnp.tile(
                flat_obs_pattern, jnp.stack([num_reps])
            )
            targ_pattern = jnp.reshape(flat_targ_pattern, targ_pattern_shape)

            # Expand the obs_pattern to have two extra channels for flags.
            # Concatenate start flag and num_reps flag to the sequence.
            obs_flag_channel_pad = jnp.zeros([sub_seq_len, 2])
            obs_start_flag = jax.nn.one_hot(
                [start_end_flag_idx], full_obs_size
            )
            num_reps_flag = jax.nn.one_hot(
                [num_repeats_channel_idx], full_obs_size
            )

            # note the concatenation dimensions.
            obs = jnp.concatenate([obs_pattern, obs_flag_channel_pad], 1)
            obs = jnp.concatenate([obs_start_flag, obs], 0)
            obs = jnp.concatenate([obs, num_reps_flag], 0)

            # Now do the same for the targ_pattern (it only has one extra
            # channel).
            targ_flag_channel_pad = jnp.zeros([sub_seq_len * num_reps, 1])
            targ_end_flag = jax.nn.one_hot(
                [start_end_flag_idx], full_targ_size
            )
            targ = jnp.concatenate([targ_pattern, targ_flag_channel_pad], 1)
            targ = jnp.concatenate([targ, targ_end_flag], 0)

            # Concatenate zeros at end of obs and begining of targ.
            # This aligns them s.t. the target begins as soon as the obs ends.
            obs_end_pad = jnp.zeros(
                [sub_seq_len * num_reps + 1, full_obs_size]
            )
            targ_start_pad = jnp.zeros([sub_seq_len + 2, full_targ_size])

            # The mask is zero during the obs and one during the targ.
            mask_off = jnp.zeros([sub_seq_len + 2])
            mask_on = jnp.ones([sub_seq_len * num_reps + 1])

            obs = jnp.concatenate([obs, obs_end_pad], 0)
            targ = jnp.concatenate([targ_start_pad, targ], 0)
            mask = jnp.concatenate([mask_off, mask_on], 0)

            obs_tensors = obs
            targ_tensors = targ
            mask_tensors = mask

            # End the loop over batch index.
            # Compute how much zero padding is needed to make tensors sequences
            # the same length for all batch elements.
            residual_obs_pad = jnp.zeros(
                [residual_length_batch[sample_idx], full_obs_size]
            )
            residual_targ_pad = jnp.zeros(
                [residual_length_batch[sample_idx], full_targ_size]
            )
            residual_mask_pad = jnp.zeros([residual_length_batch[sample_idx]])

            # Concatenate the pad to each batch element.
            obs = jnp.concatenate([obs_tensors, residual_obs_pad], 0)
            targ = jnp.concatenate([targ_tensors, residual_targ_pad], 0)
            mask = jnp.concatenate([mask_tensors, residual_mask_pad], 0)

            yield str(sample_idx), {
                "observations": obs,
                "target": targ,
                "mask": mask,
            }
