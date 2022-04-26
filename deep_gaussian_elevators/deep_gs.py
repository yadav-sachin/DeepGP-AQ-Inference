# %%
import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood

# %%
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

# %%
import urllib.request
import os
from scipy.io import loadmat
from math import floor

# %%
base_path = os.path.dirname(__file__)


class Config:
    device = "cpu"


torch.set_num_threads(15)

# %%
device = torch.device(Config.device)

# %% [markdown]
# ## Load Dataset

# %%
data = torch.tensor(
    loadmat(base_path + "/elevators.mat")["data"], device=device
).float()

# %% [markdown]
# Total Points = 16599
# Number of features = 19

# %% [markdown]
# Normalizing features data between -1 and 1

# %%

X = data[:, :-1]
X = X - X.min(axis=0).values
X = 2 * (X / X.max(axis=0).values) - 1

y = data[:, -1]

# %%
n_train = int(0.8 * len(X))
X_train = X[:n_train].contiguous()
X_test = X[n_train:].contiguous()

y_train = y[:n_train].contiguous()
y_test = y[n_train:].contiguous()

# %%
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# %%
class ToyDeepGPHiddenLayer(DeepGPLayer):
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
class DeepGP(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=train_x_shape[-1], output_dims=8, mean_type="linear"
        )

        second_hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=8, output_dims=4, mean_type="linear"
        )

        third_hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=4, output_dims=2, mean_type="linear"
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=third_hidden_layer.output_dims,
            output_dims=None,
            mean_type="constant",
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.second_hidden_layer = second_hidden_layer
        self.third_hidden_layer = third_hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        z = self.hidden_layer(inputs)
        z = self.second_hidden_layer(z)
        z = self.third_hidden_layer(z)
        output = self.last_layer(z)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            full_x_test = test_loader.dataset[:][0]
            preds = self.likelihood(self(full_x_test))
        return preds

        #     mus = []
        #     variances = []
        #     lls = []
        #     for x_batch, y_batch in test_loader:
        #         preds = self.likelihood(self(x_batch))
        #         mus.append(preds.mean)
        #         variances.append(preds.variance)
        #         lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

        # return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


# %%
model = DeepGP(train_x_shape=X_train.shape).to(device)

# %%
num_epochs = 300
num_samples = 7

# %%
optimizer = torch.optim.Adam(
    [
        {"params": model.parameters()},
    ],
    lr=0.01,
)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, X_train.shape[-2]))

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        with gpytorch.settings.num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

            minibatch_iter.set_postfix(loss=loss.item())

    if i % 2 == 0:
        print("#" * 5 + f"{i}")
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=1024)

        model.eval()
        observed_pred = model.predict(test_loader)

        rmse = torch.mean(torch.pow(observed_pred.mean - y_test, 2)).sqrt()
        print(f"RMSE: {rmse.item()}")

        msll = gpytorch.metrics.mean_standardized_log_loss(observed_pred, y_test)
        print(f"MSLL: {msll.mean().item()}")

        nlpd = gpytorch.metrics.negative_log_predictive_density(observed_pred, y_test)
        print(f"NLPD: {nlpd.mean().item()}")

        coverage_error = gpytorch.metrics.quantile_coverage_error(observed_pred, y_test)
        print(f"coverage error: {coverage_error.mean().item()}")

        model.train()

# %%
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

# %%
import math


test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1024)

model.eval()
predictive_means, predictive_variances, test_lls = model.predict(test_loader)

rmse = torch.mean(torch.pow(predictive_means.mean(0) - y_test, 2)).sqrt()
print(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")
