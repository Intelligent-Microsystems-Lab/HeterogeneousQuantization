import os
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
import pandas as pd


class FalseDict(object):
    def __getitem__(self, key):
        return 0

    def __contains__(self, key):
        return True


full_data = pd.DataFrame()
for subdir, dirs, files in os.walk("/Users/clemens/Desktop/25points"):
    for file in files:
        if "events.out.tfevents" in file:
            try:
                e_acc = EventAccumulator(
                    os.path.join(subdir, file), size_guidance=FalseDict()
                ).Reload()
                full_data[subdir.split("/")[-1]] = pd.Series(
                    [
                        tf.make_ndarray(x.tensor_proto)
                        for x in e_acc.Tensors("eval_accuracy")
                    ]
                ).sort_values(ascending=False)
            except:
                print(os.path.join(subdir, file))

full_data.to_csv("/Users/clemens/Desktop/joshi_grant.csv")
