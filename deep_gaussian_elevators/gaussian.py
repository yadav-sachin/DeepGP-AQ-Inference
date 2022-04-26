# %%
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from scipy.io import loadmat

# %% [markdown]
# Data

# %%
data = torch.tensor(loadmat("./elevators.mat")["data"]).float()
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
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(X_train, y_train, likelihood)

# %%
import os

# training_iter = 50
# training_iter = 500
training_iter = 500


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.1
)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

train_losses = []
lengthscales_list = []
noises_list = []

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(X_train)
    # Calc loss and backprop gradients
    loss = -mll(output, y_train)
    loss.backward()
    train_losses.append(loss.detach().clone().item())
    lengthscales_list.append(
        model.covar_module.base_kernel.lengthscale.detach().clone()
    )
    noises_list.append(model.likelihood.noise.item())
    print(
        "Iter %d/%d - Loss: %.3f noise: %.3f"
        % (i + 1, training_iter, loss.item(), model.likelihood.noise.item())
    )
    optimizer.step()

    if i % 10 == 0:
        print("#" * 5 + f"{i}")
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(X_test))

        rmse = torch.mean(torch.pow(observed_pred.mean - y_test, 2)).sqrt()
        print(f"RMSE: {rmse.item()}")

        msll = gpytorch.metrics.mean_standardized_log_loss(observed_pred, y_test)
        print(f"MSLL: {msll.item()}")

        nlpd = gpytorch.metrics.negative_log_predictive_density(observed_pred, y_test)
        print(f"NLPD: {nlpd.item()}")

        coverage_error = gpytorch.metrics.quantile_coverage_error(observed_pred, y_test)
        print(f"coverage error: {coverage_error.item()}")

        model.train()
        likelihood.train()

# %%
coverage_error = gpytorch.metrics.quantile_coverage_error(observed_pred, y_test)
print(f"coverage error: {coverage_error.item()}")


# %%
plt.plot(train_losses)
plt.savefig("gaussian_losses.png")
plt.close()

# %%
plt.plot(noises_list)
plt.savefig("gaussian_noises.png")
plt.close()

# %%
fig = plt.figure(figsize=(10, 8))
for i in range(len(lengthscales_list[0][0])):
    plt.plot([x[0][i] for x in lengthscales_list], label=f"{i}")
plt.legend()
plt.xlabel("epochs")
plt.savefig("gaussian_lengthscales.png")
plt.close()

# %%
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(X_test))

# %%
rmse = torch.mean(torch.pow(observed_pred.mean - y_test, 2)).sqrt()
print(f"RMSE: {rmse.item()}")

# %%
msll = gpytorch.metrics.mean_standardized_log_loss(observed_pred, y_test)
print(f"MSLL: {msll.item()}")

# %%
nlpd = gpytorch.metrics.negative_log_predictive_density(observed_pred, y_test)
print(f"NLPD: {nlpd.item()}")

# %%
coverage_error = gpytorch.metrics.quantile_coverage_error(observed_pred, y_test)
print(f"coverage error: {coverage_error.item()}")
