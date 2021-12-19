


class SpikingBlock():

  @nn.compact
  def __call__(self,
               inputs: Array,
               rng: Any,
               train: bool = True,
               survival_prob: float = None) -> Array:

  return x



class SpikingResidualBlock():




  @nn.compact
  def __call__(self,
               inputs: Array,
               rng: Any,
               train: bool = True,
               survival_prob: float = None) -> Array:



  return x




class SpikingConvNet(nn.Module):
  """EfficientNet."""
  width_coefficient: float
  depth_coefficient: float
  dropout_rate: float
  num_classes: int
  dtype: Any = jnp.float32
  act: Callable = jax.nn.relu6
  config: dict = ml_collections.FrozenConfigDict({})

  @nn.compact
  def __call__(self, x: Array, train: bool = True, rng: Any = None) -> Array:

    # jax scan function


    # Default parameters from efficientnet lite builder
    global_params = GlobalParams(
        blocks_args=_DEFAULT_BLOCKS_ARGS,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=self.config.batch_norm_epsilon if hasattr(
            self.config, 'batch_norm_epsilon') else 1e-3,
        dropout_rate=self.dropout_rate,
        survival_prob=.8,
        width_coefficient=self.width_coefficient,
        depth_coefficient=self.depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        clip_projection_output=False,
        fix_head_stem=True,  # Don't scale stem and head.
        local_pooling=True,  # special cases for tflite issues.
        use_se=False)

    _blocks_args = BlockDecoder().decode(global_params.blocks_args)

    if self.config.quant.bits is None:
      conv = partial(fake_quant_conv, dtype=self.dtype,
                     g_scale=self.config.quant.g_scale)
    else:
      conv = partial(QuantConv, dtype=self.dtype,
                     g_scale=self.config.quant.g_scale)
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=global_params.batch_norm_momentum,
                   epsilon=global_params.batch_norm_epsilon,
                   dtype=self.dtype,
                   use_bias=True,
                   use_scale=True)
    rfliters = partial(round_filters,
                       width_coefficient=global_params.width_coefficient,
                       depth_divisor=global_params.depth_divisor,
                       min_depth=global_params.min_depth,)
    conv_block = partial(
        MBConvBlock,
        clip_projection_output=global_params.clip_projection_output,
        config=self.config.quant.mbconv, bits=self.config.quant.bits)

    # Stem part.
    x = conv(
        features=rfliters(32, skip=global_params.fix_head_stem),
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='SAME',
        kernel_init=conv_kernel_initializer(),
        use_bias=False,
        name='stem_conv',
        config=self.config.quant.stem,
        bits=self.config.quant.bits,
        quant_act_sign=True,)(x)

    x = norm(name='stem_bn')(x)
    x = self.act(x)

    logging.info('Built stem layers with output shape: %s', x.shape)

    # Builds blocks.
    idx = 0
    total_num_blocks = np.sum([x.num_repeat for x in _blocks_args])
    for i, block_args in enumerate(_blocks_args):
      assert block_args.num_repeat > 0
      assert block_args.space2depth in [0, 1, 2]

      # Update block input and output filters based on depth multiplier.
      input_filters = rfliters(block_args.input_filters)
      output_filters = rfliters(block_args.output_filters)

      if (i == 0 or i == len(_blocks_args) - 1):
        repeats = block_args.num_repeat
      else:
        repeats = round_repeats(block_args.num_repeat, self.depth_coefficient)

      block_args = block_args._replace(
          input_filters=input_filters,
          output_filters=output_filters,
          num_repeat=repeats)

      survival_prob = get_survival_prob(
          global_params.survival_prob, idx, total_num_blocks)
      rng, prng = jax.random.split(rng, 2)
      x = conv_block(conv=conv,
                     norm=norm,
                     expand_ratio=block_args.expand_ratio,
                     input_filters=block_args.input_filters,
                     kernel_size=block_args.kernel_size,
                     strides=block_args.strides,
                     output_filters=block_args.output_filters,
                     id_skip=block_args.id_skip,
                     act=self.act,
                     block_num=idx)(x, train=train,
                                    survival_prob=survival_prob, rng=prng)
      idx += 1
    
      if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
        # pylint: disable=protected-access
        block_args = block_args._replace(
            input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
      for _ in range(block_args.num_repeat - 1):
        survival_prob = get_survival_prob(
            global_params.survival_prob, idx, total_num_blocks)
        rng, prng = jax.random.split(rng, 2)
        x = conv_block(conv=conv,
                       norm=norm,
                       expand_ratio=block_args.expand_ratio,
                       input_filters=block_args.input_filters,
                       kernel_size=block_args.kernel_size,
                       strides=block_args.strides,
                       output_filters=block_args.output_filters,
                       id_skip=block_args.id_skip,
                       act=self.act,
                       block_num=idx)(x, train=train,
                                      survival_prob=survival_prob, rng=prng)
        idx += 1

    # Head part.
    x = conv(features=rfliters(1280, skip=global_params.fix_head_stem),
             kernel_size=(1, 1),
             strides=(1, 1),
             padding='SAME',
             kernel_init=conv_kernel_initializer(),
             use_bias=False,
             name='head_conv',
             config=self.config.quant.head,
             bits=self.config.quant.bits,
             quant_act_sign=True)(x)
    x = norm(name='head_bn')(x)
    x = self.act(x)


    x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
    x = nn.Dense(self.num_classes,
                 kernel_init=dense_kernel_initializer(),
                 dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)

    return x


SpikingConvNet0 = partial(SpikingConvNet, width_coefficient=1.0,
                         depth_coefficient=1.0,
                         dropout_rate=0.2)
SpikingConvNet1 = partial(SpikingConvNet, width_coefficient=1.0,
                         depth_coefficient=1.1,
                         dropout_rate=0.2)
SpikingConvNet2 = partial(SpikingConvNet, width_coefficient=1.1,
                         depth_coefficient=1.2,
                         dropout_rate=0.3)
SpikingConvNet3 = partial(SpikingConvNet, width_coefficient=1.2,
                         depth_coefficient=1.4,
                         dropout_rate=0.3)
SpikingConvNet4 = partial(SpikingConvNet, width_coefficient=1.4,
                         depth_coefficient=1.8,
                         dropout_rate=0.3)
