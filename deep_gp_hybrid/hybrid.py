# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
tf.keras.backend.set_floatx("float64")  # we want to carry out GP calculations in 64 bit
tf.get_logger().setLevel("INFO")


tf.random.set_seed(42)
# %%
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--num_inducing', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--fold', type=int)
parser.add_argument('--kernel', type=str)

args = parser.parse_args()
# %%
class Config:
    num_inducing=args.num_inducing
    fold = args.fold
    # kernel = 'maternXrbf'
    kernel = args.kernel

    # num_layers=20


    # Training
    # batch_size=24
    learning_rate=args.lr
    epochs = args.epochs
    

# %%
from sklearn.preprocessing import StandardScaler
import os
script_dir = os.path.dirname(__file__)
# %%
def return_data(fold,month,with_scaling, station_id = None):
  train_input = pd.read_csv(script_dir + '/../data/beijing-18/time_feature/'+'/fold'+str(fold)+'/train_data_'+month+'_nsgp.csv.gz')
  test_input = pd.read_csv(script_dir + '/../data/beijing-18/time_feature'+'/fold'+str(fold)+'/test_data_'+month+'_nsgp.csv.gz')
  if station_id != None:
    test_input = test_input[test_input['station_id'] == station_id]
  #     test_input = test_input[test_input['station_id' == ]]
  test_output = np.array(test_input['PM25_Concentration'])
  train_output = np.array(train_input['PM25_Concentration'])
  train_input= train_input.drop(['station_id','PM25_Concentration','time','filled'],axis=1)
  try:
    test_input= test_input.drop(['PM25_Concentration','station_id','time','filled'],axis=1)
  except:
    test_input= test_input.drop(['station_id','time','filled'],axis=1)
  #     test_output= test_output.drop(['time'],axis=1)
  if with_scaling:
    scaler = StandardScaler().fit(train_input)
    train_input = pd.DataFrame(scaler.transform(train_input),columns=list(train_input.columns))
    test_input = pd.DataFrame(scaler.transform(test_input),columns=list(test_input.columns))
  return train_input,train_output,test_input,test_output


# %%
# !pip install gpflux

# %% [markdown]
# # Exp 2

# %%
import gpflow
import gpflux

# %%
for fold in [Config.fold]:
    train_input,train_output,test_input,test_output = return_data(fold=fold,month='mar',with_scaling=True)
    train_output = train_output.reshape(-1,1)
    print(train_input.shape,train_output.shape)
   
    print("Fold: ",fold)
    print("Data received")

# %%
def cont_kernel(kernel):     #,ard_num_dims, active_dims
    if kernel =='rbf':
        gp_kernel = gpflow.kernels.RBF()
    elif kernel =='matern12':
        gp_kernel = gpflow.kernels.Matern12()
    elif kernel =='matern32':
        gp_kernel = gpflow.kernels.Matern32()
    elif kernel =='matern52':
        gp_kernel = gpflow.kernels.Matern52()
    elif kernel =='matern_rbf':
        gp_kernel = gpflow.kernels.RBF() + gpflow.kernels.Matern52()
    elif kernel =='maternXrbf':
        gp_kernel = gpflow.kernels.RBF()*gpflow.kernels.Matern52()
    else:
        print("Kernel Not Found")
        exit()
    return gp_kernel

# %%
num_data = len(train_input)
num_inducing = Config.num_inducing # orig: 10
output_dim = train_output.shape[1]

kernel = cont_kernel(Config.kernel) #,train_input.shape[1], None
inducing_variable = gpflow.inducing_variables.InducingPoints(
    np.linspace(train_input.min(), train_input.max(), num_inducing).reshape(-1, 1)
)
gp_layer = gpflux.layers.GPLayer(
    kernel, inducing_variable, num_data=num_data, num_latent_gps=output_dim
)

# %%
likelihood = gpflow.likelihoods.Gaussian(0.1)

# So that Keras can track the likelihood variance, we need to provide the likelihood as part of a "dummy" layer:
likelihood_container = gpflux.layers.TrackableLayer()
likelihood_container.likelihood = likelihood

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation="linear"),
        gp_layer,
        gp_layer,
        gp_layer,
        likelihood_container,  # no-op, for discovering trainable likelihood parameters
    ]
)
loss = gpflux.losses.LikelihoodLoss(likelihood)

# %%
# opt = tf.keras.optimizers.Adam(learning_rate=Config.learning_rate)
model.compile(loss=loss, optimizer="adam")
hist = model.fit(train_input, train_output, epochs=Config.epochs, verbose=0)
plt.plot(hist.history["loss"])

# %%
test_pred = model.predict(test_input)
rmse = mean_squared_error(test_pred, test_output, squared=False)
mae = mean_absolute_error(test_pred, test_output)
r2 = r2_score(test_pred, test_output)

# %%
print(rmse, mae, r2)

# %%
## 1 gp layer 3 dense layers RBF kernel
## 100 epochs: 37.66326576265719
## 150 epochs: 36.696744913436035
## 200 epochs: 36.8858337498935


