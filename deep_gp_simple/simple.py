# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import os
import torch
import warnings

warnings.filterwarnings("ignore")

tf.keras.backend.set_floatx("float64")  # we want to carry out GP calculations in 64 bit
tf.get_logger().setLevel("INFO")
script_dir = os.path.dirname(__file__)

from keras import backend as K

tf.compat.v1.keras.backend.set_session(
    tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=4, inter_op_parallelism_threads=4
        )
    )
)

# %%
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--num_inducing", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--num_layers", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--fold", type=int)

args = parser.parse_args()
# print(args)
#%%
class Config:
    fold = args.fold
    num_inducing = args.num_inducing
    inner_layer_qsqrt_factor = 1e-5
    likelihood_noise_variance = 1e-2
    whiten = True

    num_layers = args.num_layers

    # Training
    batch_size = 256
    learning_rate = args.lr
    epochs = args.epochs


# %%
from sklearn.preprocessing import StandardScaler

# %%
def return_data(fold, month, with_scaling, station_id=None):
    train_input = pd.read_csv(
        script_dir
        + "/../data/beijing-18/time_feature/"
        + "/fold"
        + str(fold)
        + "/train_data_"
        + month
        + "_nsgp.csv.gz"
    )
    test_input = pd.read_csv(
        script_dir
        + "/../data/beijing-18/time_feature"
        + "/fold"
        + str(fold)
        + "/test_data_"
        + month
        + "_nsgp.csv.gz"
    )
    if station_id != None:
        test_input = test_input[test_input["station_id"] == station_id]
    #     test_input = test_input[test_input['station_id' == ]]
    test_output = np.array(test_input["PM25_Concentration"])
    train_output = np.array(train_input["PM25_Concentration"])
    train_input = train_input.drop(
        ["station_id", "PM25_Concentration", "time", "filled"], axis=1
    )
    try:
        test_input = test_input.drop(
            ["PM25_Concentration", "station_id", "time", "filled"], axis=1
        )
    except:
        test_input = test_input.drop(["station_id", "time", "filled"], axis=1)
    #     test_output= test_output.drop(['time'],axis=1)
    if with_scaling:
        scaler = StandardScaler().fit(train_input)
        train_input = pd.DataFrame(
            scaler.transform(train_input), columns=list(train_input.columns)
        )
        test_input = pd.DataFrame(
            scaler.transform(test_input), columns=list(test_input.columns)
        )
    return train_input, train_output, test_input, test_output


# %%
import gpflux

from gpflux.architectures import build_constant_input_dim_deep_gp
from gpflux.models import DeepGP

for fold in [Config.fold]:
    train_input, train_output, test_input, test_output = return_data(
        fold=fold, month="mar", with_scaling=True
    )
    train_output = train_output.reshape(-1, 1)
    # print(train_input.shape,train_output.shape)

    # print("Fold: ",fold)
    # print("Data received")

    config = gpflux.architectures.Config(
        num_inducing=Config.num_inducing,
        inner_layer_qsqrt_factor=Config.inner_layer_qsqrt_factor,
        likelihood_noise_variance=Config.likelihood_noise_variance,
        whiten=True,
    )
    deep_gp: DeepGP = build_constant_input_dim_deep_gp(
        train_input, num_layers=Config.num_layers, config=config
    )

    training_model: tf.keras.Model = deep_gp.as_training_model()

    # Following the Keras procedure we need to compile and pass a optimizer,
    # before fitting the model to data
    training_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=Config.learning_rate)
    )

    callbacks = [
        # Create callback that reduces the learning rate every time the ELBO plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            "loss", factor=0.95, patience=3, min_lr=1e-6, verbose=0
        ),
        # Create a callback that writes logs (e.g., hyperparameters, KLs, etc.) to TensorBoard
        gpflux.callbacks.TensorBoard(),
        # Create a callback that saves the model's weights
        tf.keras.callbacks.ModelCheckpoint(
            filepath="ckpts/", save_weights_only=True, verbose=0
        ),
    ]

    history = training_model.fit(
        {"inputs": train_input, "targets": train_output},
        batch_size=Config.batch_size,
        epochs=Config.epochs,
        callbacks=callbacks,
        verbose=0,
    )

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
ax1.plot(history.history["loss"])
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Objective = neg. ELBO")

ax2.plot(history.history["lr"])
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Learning rate")


prediction_model = deep_gp.as_prediction_model()
# print(prediction_model)
# plot(prediction_model,test_input,test_output)

# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

Test_pred = prediction_model.predict(np.array(test_input))
rmse = mean_squared_error(Test_pred, test_output, squared=False)
mae = mean_absolute_error(Test_pred, test_output)
r2 = r2_score(Test_pred, test_output)


# %%
print(rmse, mae, r2)

# %%
