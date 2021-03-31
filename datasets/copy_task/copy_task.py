"""copy_task dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import jax
import jax.numpy as jnp

# TODO(copy_task): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(copy_task): BibTeX citation
_CITATION = """
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

    obs_batch = data.observations
    targ_batch = data.target

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


class CopyTask(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for copy_task dataset."""

    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    T: int = 10
    NUM_BITS: int = 4
    MIN_LENGTH: int = 1
    MIN_REPEATS: int = 1
    MAX_REPEATS: int = 2
    SEED: int = 42
    VERSION = tfds.core.Version("1." + str(NUM_BITS) + "." + str(T))

    _norm_max: int = 10

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(copy_task): Specifies the tfds.core.DatasetInfo object

        max_length = (self.T + 1) * (self.MAX_REPEATS + 2) + 3
        # self.VERSION = tfds.core.Version('1.0.'+str(self.T))

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "observations": tfds.features.Tensor(
                        shape=(max_length, self.NUM_BITS + 2), dtype=tf.int16
                    ),
                    "target": tfds.features.Tensor(
                        shape=(max_length, self.NUM_BITS + 1), dtype=tf.int16
                    ),
                    "mask": tfds.features.Tensor(
                        shape=(max_length,), dtype=tf.int16
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://github.com/deepmind/dnc/blob/master/dnc/repeat_copy.py",
            citation=_CITATION,
        )

    def _normalise(self, val):
        return val / self._norm_max

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(copy_task): Downloads the data and defines the splits
        # path = dl_manager.download_and_extract('https://todo-data-url')

        # TODO(copy_task): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(),
        }

    def _generate_examples(self):
        # short-hand for private fields.
        min_length, max_length = self.MIN_LENGTH, self.T
        min_reps, max_reps = self.MIN_REPEATS, self.MAX_REPEATS
        num_bits = self.NUM_BITS
        batch_size = 1

        # We reserve one dimension for the num-repeats and one for the
        # start-marker.
        full_obs_size = num_bits + 2
        # We reserve one target dimension for the end-marker.
        full_targ_size = num_bits + 1
        start_end_flag_idx = full_obs_size - 2
        num_repeats_channel_idx = full_obs_size - 1

        for ds_size in range(2_000_000):

            # Samples each batch index's sequence length and the number of repeats.
            self.rng = jax.random.PRNGKey(self.SEED)
            self.rng, input_rng = jax.random.split(self.rng)
            sub_seq_length_batch = jnp.array(
                jax.random.uniform(
                    input_rng,
                    [batch_size],
                    minval=min_length,
                    maxval=max_length + 1,
                ),
                dtype=jnp.int32,
            )
            self.rng, input_rng = jax.random.split(self.rng)
            num_repeats_batch = jnp.array(
                jax.random.uniform(
                    input_rng,
                    [batch_size],
                    minval=min_reps,
                    maxval=max_reps + 1,
                ),
                dtype=jnp.int32,
            )
            # Pads all the batches to have the same total sequence length.
            total_length_batch = (
                sub_seq_length_batch * (num_repeats_batch + 1) + 3
            )

            # fixed_length = (max_length + 1) * (num_repeats_batch + 1) + 3
            max_length_batch = int((max_length + 1) * ((max_reps + 1) + 1) + 3)
            residual_length_batch = max_length_batch - total_length_batch

            obs_batch_shape = [max_length_batch, batch_size, full_obs_size]
            targ_batch_shape = [max_length_batch, batch_size, full_targ_size]
            mask_batch_trans_shape = [batch_size, max_length_batch]

            obs_tensors = []
            targ_tensors = []
            mask_tensors = []

            # Generates patterns for each batch element independently.
            for batch_index in range(batch_size):
                sub_seq_len = sub_seq_length_batch[batch_index]
                num_reps = num_repeats_batch[batch_index]

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
                targ_pattern = jnp.reshape(
                    flat_targ_pattern, targ_pattern_shape
                )

                # Expand the obs_pattern to have two extra channels for flags.
                # Concatenate start flag and num_reps flag to the sequence.
                obs_flag_channel_pad = jnp.zeros([sub_seq_len, 2])
                obs_start_flag = jax.nn.one_hot(
                    [start_end_flag_idx], full_obs_size
                )
                num_reps_flag = (
                    jax.nn.one_hot(
                        [num_repeats_channel_idx],
                        full_obs_size,
                    )
                    * self._normalise(num_reps)
                )

                # note the concatenation dimensions.
                obs = jnp.concatenate([obs_pattern, obs_flag_channel_pad], 1)
                obs = jnp.concatenate([obs_start_flag, obs], 0)
                obs = jnp.concatenate([obs, num_reps_flag], 0)

                # Now do the same for the targ_pattern (it only has one extra
                # channel).
                targ_flag_channel_pad = jnp.zeros([sub_seq_len * num_reps, 1])
                targ_end_flag = jax.nn.one_hot(
                    [start_end_flag_idx],
                    full_targ_size,
                )
                targ = jnp.concatenate(
                    [targ_pattern, targ_flag_channel_pad], 1
                )
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

                obs_tensors.append(obs)
                targ_tensors.append(targ)
                mask_tensors.append(mask)

            # End the loop over batch index.
            # Compute how much zero padding is needed to make tensors sequences
            # the same length for all batch elements.
            residual_obs_pad = [
                jnp.zeros([residual_length_batch[i], full_obs_size])
                for i in range(batch_size)
            ]
            residual_targ_pad = [
                jnp.zeros([residual_length_batch[i], full_targ_size])
                for i in range(batch_size)
            ]
            residual_mask_pad = [
                jnp.zeros([residual_length_batch[i]])
                for i in range(batch_size)
            ]

            # Concatenate the pad to each batch element.
            obs_tensors = [
                jnp.concatenate([o, p], 0)
                for o, p in zip(obs_tensors, residual_obs_pad)
            ]
            targ_tensors = [
                jnp.concatenate([t, p], 0)
                for t, p in zip(targ_tensors, residual_targ_pad)
            ]
            mask_tensors = [
                jnp.concatenate([m, p], 0)
                for m, p in zip(mask_tensors, residual_mask_pad)
            ]
            # Concatenate each batch element into a single tensor.
            # import pdb; pdb.set_trace()
            obs = jnp.reshape(jnp.concatenate(obs_tensors, 1), obs_batch_shape)
            targ = jnp.reshape(
                jnp.concatenate(targ_tensors, 1), targ_batch_shape
            )
            mask = jnp.transpose(
                jnp.reshape(
                    jnp.concatenate(mask_tensors, 0), mask_batch_trans_shape
                )
            )

            # modification axis -> [batch, time, spatial]
            obs = jnp.squeeze(jnp.moveaxis(obs, [0, 1, 2], [1, 0, 2]), axis=0)
            targ = jnp.squeeze(
                jnp.moveaxis(targ, [0, 1, 2], [1, 0, 2]), axis=0
            )
            mask = jnp.squeeze(jnp.moveaxis(mask, [0, 1], [1, 0]), axis=0)

            yield str(batch_index), {
                "observations": obs,
                "target": targ,
                "mask": mask,
            }
