#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# pylint: disable=consider-using-f-string
# pylint: disable = unrecognized-option
"""
Linear and Multiple Linear Regression, Simple neural network model
and Recursive neural network model.
In order to choose the model to perform, uncomment the function in main.
The commands are:
python neural_testing.py --type=binomial --gen=<number of generators>
--dist=<degree distribuition>--dataset_type=<Dataset>
python neural_testing.py --type=toric --n_variables=<number of target variables>
--bound=<bound of positive total degree> --dataset_type=<Dataset>
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from tensorflow.python.keras import Sequential, layers
from tensorflow.keras.layers import Dense
import tensorflow
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizer_v2.adam import Adam
from sklearn import linear_model, metrics, model_selection
from absl import logging, app, flags
from sequence_processing_batch import SequenceTrain

FLAGS = flags.FLAGS
flags.DEFINE_string("type", None, "Ideal type: toric or binomial.")
flags.mark_flag_as_required("type")

flags.DEFINE_string("gen", None, "Number of generators in each ideal.")

flags.DEFINE_string("dist", None, "Degree distribuition: weighted or "
                    "uniform")

flags.DEFINE_string("n_variables", None, "Number of target variables.")

flags.DEFINE_string("bound", None, "Positive bound on the positive degree")

flags.DEFINE_string(
    "dataset_type", None, "Dataset to work with: All; MMMSDDeg; Degree; Dim; "
    "Regularity")
flags.mark_flag_as_required("dataset_type")

flags.DEFINE_string(
    "model", None, "Linear regression: lr, simple neural"
    "network: nn, "
    "recursive neural network: rnn.")
flags.mark_flag_as_required("model")


def dataset(ideal_type, dataset_type, path_to_files_stats, path_to_files_info,
            path_to_files_degree, path_to_files_stats_degree):
    """This function builds de dataset one wish to work with
    Parameters considered are:
                -maximum degree from each ideal;
                -minimum degree from each ideal;
                -mean_degree;
                -standard deviation;
                -pure powers for binomial ideals;
                -number of generators for toric ideals;
                -ideal degree;
                -Krull dimension;
                -ideal degree;
                -regularity;
    The possible different datasets are:
                -All: contains all the parameters;
                -MMMSDDeg:maximum degree, minimum degree, mean_degree,
                standard deviation;
                -Degree: ideal degree;
                -Dim: Krull dimension;
                -Regularity: regularity;
                -PurePowers: pure powers in case of binomial ideals
                 or number of generators for toric ideals;
    Args:
        -dataset_type: define the dataset to build;
        -joined_files_degrees: the CSV files to extract information
        of the ideals exponents;
        -path_to_files_stats: the CSV file to extract information about
        the Gröbner basis calculation of each ideal:
        polynomial additions, monomial additions and zero additions,
         and the about each ideal: krull dimension,
        ideal degree and regularity;
        -path_to_files_info: the CSV file to extract information of the
        ideals exponents information: maximum degree,
         minimum degree, mean_degree, standard deviation, pure powers for
         binomial ideals and number of generators for toric ideals.
    Returns:
        -x: dataset regarding the ideal features;
        -y: dataset with the values to predict;
    """

    accepted_strings = {
        "All", "MMMSDDeg", "Degree", "Dim", "Regularity", "PurePowers"
    }
    logging.info("Creates the dataset according to the dataset_type")
    # The features file is organized as: Degree, Dim, Regularity,
    # Maximum degree for degree type 0,  Minimum
    # degree type 0, Mean degree type 0, Std degree type 0,
    # Pure Powers/Number of generators, Maximum degree type 1,
    # Minimum degree type 1, Mean degree type 1,
    # Std degree type 1.
    # Degree type 0: sum(list(abs(np.subtract(mon1, mon2))))
    # Degree type 1: max(sum(mon1), sum(mon2))

    if dataset_type in accepted_strings:
        logging.info("Creating polynomial additions dataset")
        df_y = pd.read_csv(path_to_files_stats, engine="python")
        y = df_y.iloc[:, 1]

        logging.info("Creating features dataset")
        df = pd.read_csv(path_to_files_info, engine="python")

        if dataset_type == "All":
            # Degree_1
            x = df.iloc[:, [1, 2, 3, 9, 10, 11, 12, 8]]
            # Degree_0
            # x = df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8]]
            x = pd.DataFrame(x).values
        elif dataset_type == "MMMSDDeg":
            # Degree_1
            x = df.iloc[:, [9, 10, 11, 12]]
            # Degree_0
            # x = df.iloc[:, [4, 5, 6, 7]]
            x = pd.DataFrame(x).values
        elif dataset_type == "Degree":
            x = df.iloc[:, 1]
            x = pd.DataFrame(x).values
        elif dataset_type == "Dim":
            x = df.iloc[:, 2]
            x = pd.DataFrame(x).values
        elif dataset_type == "Regularity":
            x = df.iloc[:, 3]
            x = pd.DataFrame(x).values
        elif dataset_type == "PurePowers":
            x = df.iloc[:, 8]
            x = pd.DataFrame(x).values
    elif dataset_type == "Powers":
        logging.info("Reading polynomial additions dataset")
        df_y = pd.read_csv(path_to_files_stats_degree,
                           engine="python",
                           header=None)
        y = df_y.iloc[:, 0]
        logging.info("Reading Degree file")
        if ideal_type == "binomial":
            df = pd.read_csv(path_to_files_degree, engine="python", header=None)
            x = pd.DataFrame(df).values
        else:
            with open(path_to_files_degree, encoding="utf-8") as f:
                x = f.readlines()

    y = pd.DataFrame(y).values

    return x, y


def regr_multi(dataset_type, x, y, rgr_info_path):
    """Performs multiple linear regression or linear regression
    for polynomial addition prediction

            Args:
                dataset_type:
                x: file containing the required parameters
                corresponding the wanted model.
                y: file containing information about polynomial
                additions;
                rgr_info_path: path to save results in csv format.
            Returns:
                regr_information: returns results from the regression
                 model in CSV format, containing the
                coefficients of each considered feature, r squared,
                 mean absolute error, mean squared error and root
                mean square error.
    """
    logging.info(
        "Division of dataset X and Y into train, test and validation datasets")

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.1, random_state=0)
    logging.info("Linear regression model creation")
    regr = linear_model.LinearRegression()
    logging.info("Data fit")
    regr.fit(x_train, y_train)

    print("Intercept: \n", regr.intercept_)
    print("Coefficients: \n", regr.coef_[0])
    rounded_coeff = np.round(regr.coef_[0], 2)

    logging.info("Polynomial prediction for the test set")
    y_pred_regr = regr.predict(x_test)

    meanaberr = metrics.mean_absolute_error(y_test, y_pred_regr)
    meansqerr = metrics.mean_squared_error(y_test, y_pred_regr)
    rootmeansqerr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_regr))
    print("R squared: {:.2f}".format(regr.score(x, y) * 100))
    print("Mean Absolute Error:{:.2f}".format(meanaberr))
    print("Mean Square Error:{:.2f}".format(meansqerr))
    print("Root Mean Square Error:{:.2f}".format(rootmeansqerr))
    accepted_strings = {"Degree", "Dim", "Regularity", "PurePowers"}
    if dataset_type == "All":
        regr_information = pd.DataFrame({
            "Coeff: maximum degree": [rounded_coeff[0]],
            "Coeff: minimum degree": rounded_coeff[1],
            "Coeff: mean degree": [rounded_coeff[2]],
            "Coeff: std degree": [rounded_coeff[3]],
            "Coeff: pure power": [rounded_coeff[4]],
            "Coeff: Degree": [rounded_coeff[5]],
            "Coeff: Dim": [rounded_coeff[6]],
            "Coeff: Regularity": [rounded_coeff[7]],
            "R squared": np.round(regr.score(x, y) * 100, 2),
            "Mean Absolute Error": format(meanaberr),
            "Mean Square Error": format(meansqerr),
            "Root Mean Square Error": format(rootmeansqerr)
        })
    elif dataset_type == "MMMSDDeg":
        regr_information = pd.DataFrame({
            "Coeff: maximum degree": [rounded_coeff[0]],
            "Coeff: minimum degree": rounded_coeff[1],
            "Coeff: mean degree": [rounded_coeff[2]],
            "Coeff: std degree": [rounded_coeff[3]],
            "R squared": np.round(regr.score(x, y) * 100, 2),
            "Mean Absolute Error": format(meanaberr),
            "Mean Square Error": format(meansqerr),
            "Root Mean Square Error": format(rootmeansqerr)
        })
    elif dataset_type in accepted_strings:
        regr_information = pd.DataFrame({
            "Coeff: " + dataset_type: [rounded_coeff[0]],
            "R squared": np.round(regr.score(x, y) * 100, 2),
            "Mean Absolute Error": meanaberr,
            "Mean Square Error": meansqerr,
            "Root Mean Square Error": rootmeansqerr
        })
    else:
        print("Error dataset_type")

    regr_information.to_csv(rgr_info_path)


def simple_nn_model(x, y, nn_info_path, nn_graph_path):
    """Simple neural network model for polynomial addition prediction:
                Args:
                    x: file containing the required parameters corresponding
                     the wanted model;
                    y: file containing information about polynomial additions;
                    nn_info_path: path to save results in csv format;
                    nn_graph_path: path to save the models performance graph.
                Returns:
                    nn_information: returns results from the model training
                    in CSV format, contains mean absolute
                    error, mean squared error and the r squared.
    """
    logging.info(
        "Divide dataset X and Y into train, test and validation datasets")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.1, random_state=0)
    logging.info("Vanilla NN Model")
    model = Sequential()
    model.add(
        Dense(300,
              input_dim=x.shape[1],
              activation="relu",
              kernel_initializer="he_normal"))
    model.add(Dense(1, activation="linear"))
    optim = Adam(learning_rate=0.00001)

    logging.info("Compile the keras model")
    model.compile(loss="mse", optimizer=optim)

    es = EarlyStopping(monitor="val_loss", patience=20)
    logging.info("Fit the keras model on the dataset")
    history = model.fit(x_train,
                        y_train,
                        epochs=300,
                        batch_size=16,
                        verbose=2,
                        validation_split=0.2,
                        callbacks=[es])
    print(history.history.keys())
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(nn_graph_path)
    plt.show()
    logging.info("Evaluate on test set")
    yhat = model.predict(x_test)
    mae = metrics.mean_absolute_error(y_test, yhat)
    mse = metrics.mean_squared_error(y_test, yhat)
    rs = metrics.r2_score(y_test, yhat)
    nn_information = pd.DataFrame({
        "mean absolute error: %.2f" % mae,
        "mean squared error: %.2f" % mse,
        "r squared: %.2f" % rs
    })
    nn_information.to_csv(nn_info_path)


def rnn_model(x, y, rnn_info_path, rnn_graph_path, rnn_predict_path,
              ideal_type):
    """Recursive neural network (RNN) with gated recurrent unit(GRU) model
     for polynomial addition prediction:
                    Args:
                        x: file containing the required parameters
                         corresponding the wanted model;
                        y: file containing information about polynomial
                         additions;
                        rnn_info_path: path to save results in CSV format;
                        rnn_graph_path: path to save the model performance
                        graph;
                        rnn_predict_path: path to save the predicted values;
                        ideal_type: information about the ideals dataset.
                    Returns:
                        rnn_information: returns results from the model training
                         in CSV format, contains mean absolute
                        error, mean squared error and the r squared.
        """

    if ideal_type == "binomial":
        logging.info(
            "Divide dataset X and Y into train, test and validation datasets")
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y, test_size=0.1, random_state=0)
        print(len(x_train[0]))
        x_train = np.reshape(x_train, (x_train.shape[0], int(FLAGS.gen), 6))
        x_test = np.reshape(x_test, (x_test.shape[0], int(FLAGS.gen), 6))
        shape = (int(FLAGS.gen), 6)

        print(x_train[0])
        print(x_train.shape)
        print(y_train.shape)
        print(x_train)
        print(y_train)

        logging.info("GRU Model")
        optim = Adam(learning_rate=0.0001)
        model = Sequential()
        model.add(layers.GRU(300, activation="relu", input_shape=shape))
        model.add(layers.Dense(1, activation="linear"))
        print(model.summary())

        logging.info("Compile the keras model")
        model.compile(loss="mean_absolute_error", optimizer=optim)

        es = EarlyStopping(monitor="val_loss", patience=10)
        logging.info("Fit the keras model on the dataset")
        history = model.fit(x_train,
                            y_train,
                            epochs=300,
                            batch_size=32,
                            verbose=2,
                            validation_split=0.2,
                            callbacks=es)

        print(history.history.keys())
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.savefig(rnn_graph_path)
        plt.show()
        logging.info("Evaluate on test set")
        yhat = model.predict(x_test)
        yhat = np.array(yhat)

    else:
        logging.info(
            "Divide dataset X and Y into train, test and validation datasets")
        print(len(x))
        x_train = x[:int(len(x) * .8)]
        x_test = x[int(len(x) * .9):]
        y_train = y[:int(len(y) * .8)]
        y_test = y[int(len(y) * .9):]
        x_val = x[int(len(x) * .8):int(len(x) * .9)]
        y_val = y[int(len(y) * .8):int(len(y) * .9)]
        print("---------test---------")
        print(len(x_test))
        print(len(y_test))
        print("---------train---------")
        print(len(x_train))
        print(len(y_train))
        x_train = SequenceTrain(x_train, y_train, batch_size=20)
        x_val = SequenceTrain(x_val, y_val, batch_size=20)

        logging.info("GRU Model")
        optim = Adam(learning_rate=0.0001)
        model = Sequential()
        model.add(layers.GRU(128, activation="tanh"))
        model.add(layers.Dense(1, activation="linear"))

        logging.info("Compile the keras model")
        model.compile(loss="mean_absolute_error", optimizer=optim)

        es = EarlyStopping(monitor="val_loss",
                           patience=20,
                           restore_best_weights=True)

        logging.info("Fit the keras model on the dataset")
        history = model.fit(x_train,
                            epochs=300,
                            verbose=2,
                            validation_data=x_val,
                            callbacks=[es],
                            use_multiprocessing=True,
                            workers=4,
                            max_queue_size=16)

        print(history.history.keys())
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.savefig(rnn_graph_path)
        # plt.show()
        logging.info("Evaluate on test set")
        logging.info("Padding the x_train")
        yhat = []

        batch_test_size = 10000
        for batch_test in np.array_split(x_test, batch_test_size):
            k = []
            for line in batch_test:
                values = list(line.split(","))
                values = [
                    re.findall(r"\d+", x)[0]
                    for x in values
                    if len(re.findall(r"\d+", x)) > 0
                ]
                values = np.array(values)
                a = values.astype(np.float)
                k.append(a)
            x_test_padded = tensorflow.keras.preprocessing.sequence. \
                pad_sequences(k, padding="post", value=-10, maxlen=None)
            for line in x_test_padded:
                j = line.reshape([1, int(line.shape[0] / (16)), 16])
                y = model.predict(j)
                yhat.append((y[0])[0])

        yhat = np.array(yhat)

    mae = metrics.mean_absolute_error(y_test, yhat)
    mse = metrics.mean_squared_error(y_test, yhat)
    rs = metrics.r2_score(y_test, yhat)
    rnn_information = pd.DataFrame({
        "mean absolute error: %.3f" % mae,
        "mean squared error: %.3f" % mse,
        "r squared: %.2f" % rs
    })
    rnn_information.to_csv(rnn_info_path)
    pd.DataFrame(yhat).to_csv(rnn_predict_path)


def main(_):
    if FLAGS.type == "binomial":

        path_to_files_stats = f"data/big_data" \
                              f"/3-20-{FLAGS.gen}-{FLAGS.dist}" \
                              f"/polynomial_add_dataset.csv"

        path_to_files_info = f"data/big_data" \
                             f"/3-20-{FLAGS.gen}-{FLAGS.dist}/" \
                             f"features_dataset.csv"

        path_to_files_degree = f"data/big_data" \
                               f"/3-20-{FLAGS.gen}-{FLAGS.dist}" \
                               f"/concatenated.csv"
        path_to_files_stats_degree = f"data/big_data" \
                                     f"/3-20-{FLAGS.gen}-{FLAGS.dist}/" \
                                     f"concatenated_stats.csv"

        rgr_info_path = f"data/big_data" \
                        f"/3-20-{FLAGS.gen}-{FLAGS.dist}/info_regression" \
                        f"_{FLAGS.dataset_type}.csv"
        nn_info_path = f"data/big_data" \
                       f"/3-20-{FLAGS.gen}-{FLAGS.dist}/info_NN_" \
                       f"{FLAGS.dataset_type}.csv"
        nn_graph_path = f"data/big_data" \
                        f"/3-20-{FLAGS.gen}-{FLAGS.dist}/graph_NN_" \
                        f"{FLAGS.dataset_type}.png"
        rnn_info_path = f"data/big_data" \
                        f"/3-20-{FLAGS.gen}-{FLAGS.dist}/info_RNN_" \
                        f"{FLAGS.dataset_type}.csv"
        rnn_graph_path = f"data/big_data" \
                         f"/3-20-{FLAGS.gen}-{FLAGS.dist}/graph_RNN_" \
                         f"{FLAGS.dataset_type}.png"

        rnn_predict_path = f"data/big_data" \
                           f"/3-20-{FLAGS.gen}-{FLAGS.dist}/predict_y_RNN_" \
                           f"{FLAGS.dataset_type}.csv"

    else:

        path_to_files_stats = f"data/big_data" \
                              f"/toric-{FLAGS.n_variables}-0-" \
                              f"{FLAGS.bound}-8/polynomial_add_dataset.csv"
        path_to_files_info = f"data/big_data" \
                             f"/toric-{FLAGS.n_variables}-0" \
                             f"-{FLAGS.bound}-8/features_dataset.csv"
        path_to_files_degree = f"data/big_data" \
                               f"/toric-{FLAGS.n_variables}-0" \
                               f"-{FLAGS.bound}-8/concatenated.csv"
        path_to_files_stats_degree = f"data/big_data" \
                                     f"/toric-{FLAGS.n_variables}-0" \
                                     f"-{FLAGS.bound}-8/concatenated_stats.csv"
        rgr_info_path = f"data/big_data" \
                        f"/toric-{FLAGS.n_variables}-0-" \
                        f"{FLAGS.bound}-8/info_regression" \
                        f"_{FLAGS.dataset_type}.csv"
        nn_info_path = f"data/big_data" \
                       f"/toric-{FLAGS.n_variables}-0" \
                       f"-{FLAGS.bound}-8/info_NN" \
                       f"_{FLAGS.dataset_type}_NG.csv"
        nn_graph_path = f"data/big_data" \
                        f"/toric-{FLAGS.n_variables}-0" \
                        f"-{FLAGS.bound}-8/graph_NN" \
                        f"_{FLAGS.dataset_type}_NG.png"
        rnn_info_path = f"data/big_data" \
                        f"/toric-{FLAGS.n_variables}-0" \
                        f"-{FLAGS.bound}-8/info_RNN" \
                        f"_{FLAGS.dataset_type}.csv"
        rnn_graph_path = f"data/big_data" \
                         f"/toric-{FLAGS.n_variables}-0" \
                         f"-{FLAGS.bound}-8/graph_RNN" \
                         f"_{FLAGS.dataset_type}.png"
        rnn_predict_path = f"data/big_data" \
                           f"/toric-{FLAGS.n_variables}-0-" \
                           f"{FLAGS.bound}-8/predict_y_RNN" \
                           f"_{FLAGS.dataset_type}.csv"
    logging.info("Files paths joined")

    ideal_type = FLAGS.type
    dataset_type = FLAGS.dataset_type
    model = FLAGS.model
    logging.info("Dataset creation")
    x, y = dataset(ideal_type, dataset_type, path_to_files_stats,
                   path_to_files_info, path_to_files_degree,
                   path_to_files_stats_degree)

    logging.info("Run wanted model")
    if model == "lr":
        regr_multi(dataset_type, x, y, rgr_info_path)
    elif model == "nn":
        simple_nn_model(x, y, nn_info_path, nn_graph_path)
    elif model == "rnn":
        rnn_model(x, y, rnn_info_path, rnn_graph_path, rnn_predict_path,
                  ideal_type)
    else:
        print("Error")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
