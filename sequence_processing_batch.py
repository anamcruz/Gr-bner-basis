# pylint: disable = consider-using-f-string
# pylint: disable = unrecognized-option
"""
Complementary class to process the exponents of toric ideals.
The program reads the datasets in a defined batch size and pads
sequences to the same length. The padding value chosen was -10.
Return sequences in the same size and returns a Tensor object.
"""

import numpy as np
import tensorflow
import re
from tensorflow.python.keras.utils import data_utils


class SequenceTrain(data_utils.Sequence):
    """
    Takes in sequence data, divides it in equal sized batches. Each batch is
    converted from string to numpy arrays and the sequences are padded into
    an equal size and reshaped to be used as input for the training of the RNN.
    It returns a Tensor object.
    """

    def __init__(self, x_data, y_data, batch_size, n_variables=8):
        # defining the input: x_data are the exponents, y_data is
        # the number of polynomial additions, batch_size is the size of
        # the sample we want to process at a time and n_variables are the total
        # number of variables.
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.n_variables = n_variables

    def __len__(self):
        # dividing the dataset length into batches sizes
        return int(np.ceil(len(self.x_data) / float(self.batch_size)))

    def __getitem__(self, idx):

        # define the batches to process, each batch is associated by an index
        batch_x = self.x_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        k = []

        # the batches are initially read as string with extra commas,
        # so in the next function we remove the extra commas and transform
        # each row to numpy arrays.
        for line in batch_x:
            # divide by commas
            values = list(line.split(","))
            # finds all integer values
            values = [
                re.findall(r"\d+", x)[0]
                for x in values
                if len(re.findall(r"\d+", x)) > 0
            ]
            # saves only integers values in numpy array
            values = np.array(values)
            # turn from string to float values
            a = values.astype(np.float)
            # saves every numpy array in a list
            k.append(a)
        # padding the new batch k with -10 values, so all elements from
        # k has the same size
        j = tensorflow.keras.preprocessing.sequence.pad_sequences(
            k, padding="post", value=-10, maxlen=None)

        # reshaping in matrix format, each row represents each binomial
        # generator
        j = j.reshape([
            self.batch_size,
            int(j.shape[-1] / (self.n_variables * 2)), self.n_variables * 2
        ])
        # j is saved as a Python object, so it returns the correspondent
        # Tensor object
        return tensorflow.convert_to_tensor(
            j, dtype="float32"), tensorflow.convert_to_tensor(batch_y,
                                                              dtype="float32")
