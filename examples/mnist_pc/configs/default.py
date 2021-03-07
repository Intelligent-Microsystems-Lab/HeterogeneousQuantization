# copied from
# https://github.com/google/flax/tree/master/examples/mnist
#
# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # learning rate
    config.learning_rate = 0.1
    # momentum for optimizer
    config.momentum = 0.9
    # number of examples per training batch
    config.batch_size = 128
    # number of total training epochs
    config.num_epochs = 10
    # number of bits used for gradients
    config.grad_bw = 4
    # number of initial epochs without quantization
    config.initial_fp_num_epochs = 0
    # log file to compare current run to
    # default tfevents file comes from running flax mnist example
    config.log_compare = (
        "logs/master/events.out.tfevents.1613448567."
        "clemenss-mbp.dhcp.nd.edu.14089.137.v2"
    )
    return config
