# %%
import torch
import tqdm
import gpytorch


from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
import argparse
import matplotlib.pyplot as plt
from probml_utils import latexify, savefig, is_latexify_enabled
import seaborn as sns
# %%
from gpytorch.constraints.constraints import GreaterThan

# %%
# from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.models import deep_gps
from gpytorch.mlls import DeepApproximateMLL

# %%
import urllib.request
import os
import pandas as pd
from scipy.io import loadmat
import numpy as np
from math import floor


class Config:
    device = "cpu"


device = torch.device(Config.device)

# %%
class Config:
    num_inducing = 100
    fold = 1
    lr = 0.05
    epochs = 222
    num_samples = 7


# %%
from sklearn.preprocessing import StandardScaler
import os

script_dir = os.path.dirname(__file__)


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
for fold in [Config.fold]:
    train_input, train_output, test_input, test_output = return_data(
        fold=fold, month="mar", with_scaling=True, station_id=1006
    )

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(
    torch.tensor(train_input.values).float().to(device),
    torch.tensor(train_output).float().to(device),
)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# %%
class ToyDeepGPHiddenLayer(deep_gps.DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type="constant"):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super(ToyDeepGPHiddenLayer, self).__init__(
            variational_strategy, input_dims, output_dims
        )

        if mean_type == "constant":
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
            ard_num_dims=None,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(
                    gpytorch.settings.num_likelihood_samples.value(), *inp.shape
                )
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


# %%
class DeepGP(deep_gps.DeepGP):
    def __init__(self, train_x_shape):
        super(DeepGP, self).__init__()
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=12,
            mean_type="linear",
            num_inducing=Config.num_inducing,
        )

        second_hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=12,
            output_dims=4,
            mean_type="linear",
            num_inducing=Config.num_inducing,
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=second_hidden_layer.output_dims,
            output_dims=None,
            mean_type="constant",
            num_inducing=Config.num_inducing,
        )

        self.hidden_layer = hidden_layer
        self.second_hidden_layer = second_hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        z = self.hidden_layer(inputs)
        z = self.second_hidden_layer(z)
        output = self.last_layer(z)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            full_x_test = test_loader.dataset[:][0]
            preds = self.likelihood(self(full_x_test))
        return preds


# %%
model = DeepGP(train_x_shape=train_input.shape).to(device)
# To load model:
# state_dict = torch.load('model_epoch221.pth')
# model = DeepGP(train_x_shape=train_input.shape).to(device)  # Create a new GP model
# model.load_state_dict(state_dict)
num_epochs = Config.epochs
num_samples = Config.num_samples
optimizer = torch.optim.Adam(
    [
        {"params": model.parameters()},
    ],
    lr=Config.lr,
)
mll = DeepApproximateMLL(
    VariationalELBO(model.likelihood, model, train_input.shape[-2])
)

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
train_losses = []

for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        with gpytorch.settings.num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            train_losses.append(loss.detach().item())
            loss.backward()
            optimizer.step()

            minibatch_iter.set_postfix(loss=loss.item())

    if i%5 == 0:
        model.eval()
        test_dataset = TensorDataset(
            torch.tensor(test_input.values).float(), torch.tensor(test_output).float()
        )
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)
        torch.save(model.state_dict(), f'model_epoch{i}.pth')
        preds = model.predict(test_loader)
        preds_stds = torch.sqrt(preds.variance).T.mean(axis=1)
        preds_mean = preds.mean.T.mean(axis=1)
        lower_line = preds_mean - 2 * preds_stds
        upper_line = preds_mean + 2 * preds_stds

        os.environ["LATEXIFY"]="1"
        os.environ["FIG_DIR"]="final_plots/"
        latexify(width_scale_factor=1.5, fig_height=2)
        plt.fill_between(
            range(len(test_output)),
            lower_line,
            upper_line,
            facecolor="gray",
            alpha=0.5,
            label="95$\%$ confidence",
        )
        plt.plot(preds_mean, label="Predicted")
        y_test = test_output
        plt.plot(test_output, label="Ground Truth")
        plt.ylabel("PM$_{2.5}$")
        plt.legend()
        plt.xlabel("Timeline, March 2015 (hours)")
        sns.despine()
        savefig("deep_variational_station")

        plt.savefig(f"deep_variational_station_1006_{i}.png")
        model.train()

# %%
test_dataset = TensorDataset(
    torch.tensor(test_input.values).float(), torch.tensor(test_output).float()
)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)
import math

model.eval()
observed_pred = model.predict(test_loader)
y_test = test_loader.dataset[:][1]

rmse = torch.mean(torch.pow(observed_pred.mean - y_test, 2)).sqrt()
print(f"RMSE: {rmse.item()}")

msll = gpytorch.metrics.mean_standardized_log_loss(observed_pred, y_test)
print(f"MSLL: {msll.mean().item()}")

nlpd = gpytorch.metrics.negative_log_predictive_density(observed_pred, y_test)
print(f"NLPD: {nlpd.mean().item()}")

coverage_error = gpytorch.metrics.quantile_coverage_error(observed_pred, y_test)
print(f"coverage error: {coverage_error.mean().item()}")
