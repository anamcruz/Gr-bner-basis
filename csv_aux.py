#!/usr/bin/python3.8
# pylint: disable = consider-using-with
# pylint: disable = unused-variable
# pylint: disable = consider-using-f-string
# pylint: disable = unrecognized-option
"""CSV information extration

The command to run the code is:
python csv_degree.py --type=toric --n_variables=<number of
target variables>
--bound=<bound of positive total degree>

python csv_degree.py --type=binomial --gen=<number of generators>
--dist=<degree distribuition>
"""

import numpy as np
import pandas as pd
import math
import glob
import os
from absl import logging
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("type", None, "Number of generators in each ideal.")
flags.mark_flag_as_required("type")

flags.DEFINE_integer("gen", None, "Number of generators in each ideal.")

flags.DEFINE_string("dist", None, "Degree distribuition: weighted or "
                    "uniform")

flags.DEFINE_integer("n_variables", None, "Number of generators in each ideal.")

flags.DEFINE_integer("bound", None, "Degree distribuition: weighted or "
                     "uniform")


def create_info(joined_files, ideal_type, info_path):
    """Generates info_degree
     Useful parameters are calculated using Degree file:
        -maximum degree from each Ideal.
        -minimum degree from each Ideal.
        -mean_degree.
        -standard deviation.
        -between others to add.

    Args:
        joined_files: all files containing the ideals exponents.
            example: a row in the file will look like this: "[9, 10,
            0, 3, 2, 0, 9, 4, 3, 0, 1, 6, 0, 8, 1, 3, 2, 3, 2, 14, 3, 0, 11, 0]"
        ideal_type: binomial or toric ideals.
        info_path: path for the new file containing metrics of the exponents.
    Returns:
        info_degree: matrix containing the values used for the multiple
        linear regression
    """

    j = 0
    for filename in joined_files:
        print(filename)
        degrees_dataset = pd.read_csv(filename, engine="python", header=None)
        a = degrees_dataset.values
        # Creates empty array for saving the information
        info_degree = np.zeros([1000, 9])
        # Creates the ID for properly saving the file
        file_associated_splited = joined_files[j].split("/")[8]
        file_associated = file_associated_splited.split(".")[0]
        print(j)
        # Info_degree contains the maximum, minimum, mean, std degree and pure
        # powers/number of generators for each ideal.
        if ideal_type == "binomial":

            degree_0 = np.zeros(FLAGS.gen)
            degree_1 = np.zeros(FLAGS.gen)
            degree_2 = np.zeros(FLAGS.gen)
            for i in range(-1, len(a)):
                b = a[i]
                pure_powers = 0
                # The chunks divide the list of degrees to separated lists
                # containing the three exponents of each individual monomial.
                # for example: "[9, 10, 0] [3, 2, 0], ..."
                chunk = [b[j:j + 3] for j in range(0, len(b), 3)]

                for k in range(FLAGS.gen):
                    mon1 = chunk[2 * k]
                    mon2 = chunk[2 * k + 1]

                    degree_0[k] = sum(abs(np.subtract(mon1, mon2)))
                    degree_1[k] = max(sum(mon1), sum(mon2))
                    degree_2[k] = min(sum(mon1), sum(mon2))
                    # degree_0, degree_1 and degree_2 are arrays containing
                    # different ways of considering each polynomial degree

                    n_zeros_1 = np.count_nonzero(mon1)

                    if n_zeros_1 == 1:
                        pure_powers = pure_powers + 1
                std_d = np.std(degree_1)
                std_a = np.std(degree_0)
                info_degree[i, 0] = max(degree_0)
                info_degree[i, 1] = min(degree_0)
                info_degree[i, 2] = sum(degree_0) / len(degree_0)
                info_degree[i, 3] = std_a
                info_degree[i, 4] = max(degree_1)
                info_degree[i, 5] = min(degree_1)
                info_degree[i, 6] = sum(degree_1) / len(degree_1)
                info_degree[i, 7] = std_d
                info_degree[i, 8] = pure_powers

        else:

            # For the toric ideal case
            for i in range(-1, len(a)):

                b = [x for x in a[i] if math.isnan(x) is False]
                pure_powers = 0
                chunk = [b[j:j + 8] for j in range(0, len(b), 8)]
                # The size of the following 3 arrays are (len(chunk)/2) since
                # the len(chunk) is the total number of monomials in each
                # ideal with 8 variables total and each generator of the
                # ideal is binomial, therefor the number of generators is (
                # len(chunk)/2).
                degree_0 = np.zeros(int(len(chunk) / 2))
                degree_1 = np.zeros(int(len(chunk) / 2))
                degree_2 = np.zeros(int(len(chunk) / 2))
                # I need to define each array (degree, degree_2, degree_0) size
                # according to
                # each row, but it
                # depends on the chunks, but then again every for-loop will turn
                # it to an array of zeros
                for k in range(int(len(chunk) / 2)):

                    mon1 = chunk[2 * k]
                    mon2 = chunk[2 * k + 1]

                    degree_0[k] = sum(list(abs(np.subtract(mon1, mon2))))
                    degree_1[k] = max(sum(mon1), sum(mon2))
                    degree_2[k] = min(sum(mon1), sum(mon2))
                    # degree_0, degree and degree_2 are arrays containing
                    # different ways of considering each polynomial degree

                std_d = np.std(degree_1)
                std_a = np.std(degree_0)
                info_degree[i, 0] = max(degree_0)
                info_degree[i, 1] = min(degree_0)
                info_degree[i, 2] = sum(degree_0) / len(degree_0)
                info_degree[i, 3] = std_a
                info_degree[i, 4] = max(degree_1)
                info_degree[i, 5] = min(degree_1)
                info_degree[i, 6] = sum(degree_1) / len(degree_1)
                info_degree[i, 7] = std_d
                info_degree[i, 8] = (len(chunk) / 2)

        # Saves the Degree info into a new file
        pd.DataFrame(info_degree).to_csv(info_path + f"{file_associated}.csv")
        j = j + 1


def main(_):
    ideal_type = FLAGS.type
    if ideal_type == "binomial":

        files = os.path.join(
            f"/home/anamaria/data_generation/data/big_data/"
            f"3-20-{str(FLAGS.gen)}"
            f"-{str(FLAGS.dist)}/Degree",
            f"3-20-{str(FLAGS.gen)}-{str(FLAGS.dist)}*.csv")


        info_path = f"/home/anamaria/data_generation/data/big_data/3-20" \
                    f"-{str(FLAGS.gen)}-" \
                    f"{str(FLAGS.dist)}/Info_degree/"

    else:
        files = os.path.join(
            f"/home/anamaria/data_generation/data/big_data/"
            f"toric-{str(FLAGS.n_variables)}-0-"
            f"{str(FLAGS.bound)}-8/Degree_test",
            f"toric-{str(FLAGS.n_variables)}-0-"
            f"{str(FLAGS.bound)}-8*.csv")
        info_path = f"/home/anamaria/data_generation/data/big_data/toric-" \
                    f"{str(FLAGS.n_variables)}-0-" \
                    f"{str(FLAGS.bound)}-8/Info_degree/"

    joined_files = glob.glob(files)

    # cleaning the Degree data, removing all '{' and '}'
    filenames = []
    for filename in joined_files:
        text_dg = open(filename, "r", encoding="utf-8")
        text_dg = "".join(list(text_dg)) \
            .replace("{", "").replace("}", "")
        x = open(filename, "w", encoding="utf-8")
        x.writelines(text_dg)
        x.close()

    # saving the filenames in a list for ahead saving the correspondent
    # information with the same filename
    for i in range(len(joined_files)):
        filenames.append(joined_files[i])
    # this for-loop serves for matching the degree lines and matching their
    # length
    for filename in filenames:
        with open(filename, "r", encoding="utf-8") as f:
            data = f.read().splitlines()
        cols = [row.count(",") + 1 for row in data]
        max_cols = max(cols)

        for index_r, row in enumerate(data):
            if cols[index_r] < max_cols:
                data[index_r] += (max_cols - cols[index_r]) * ","
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines("\n".join(data))
        f.close()

    create_info(joined_files, ideal_type, info_path)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
