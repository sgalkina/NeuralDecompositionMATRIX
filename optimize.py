import os
from loguru import logger

import numpy as np
import pandas as pd
from scipy.stats import zscore
import argparse

import matplotlib.pyplot as plt

from ND.encoder import cEncoder
from ND.decoder import Decoder
from ND.CVAE import CVAE
from ND.helpers import expand_grid
from ND.decoder_multiple_covariates import Decoder_multiple_covariates
from ND.CVAE_multiple_covariates import CVAE_multiple_covariates
from ND.utils import plot_integrals, generate_timestamp_id, plot_variance, plot_features

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


parser = argparse.ArgumentParser(description="CVAE with multiple covariates")
parser.add_argument("--n_iter", type=int, default=5000, help="Number of training iterations (default: 5000)")
parser.add_argument("--device", type=str, default="cpu", help="Device to train the model on (default: cpu)")
parser.add_argument("--columns", nargs="+", help="List of column names to use from the covariates file")
args = parser.parse_args()

n_iter = args.n_iter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
selected_columns = args.columns

logger.info(f"Start")

unique_id = generate_timestamp_id()
logger.info(f"Generated ID: {unique_id}")

result_matrix = pd.read_csv('data/normalized_features_82_samples.csv')
df_cov = pd.read_excel('data/MATRIX_TS_blocks.xlsx', sheet_name='Factors')

# Check if specific columns were provided
if selected_columns:
    logger.info(f"Using selected columns: {selected_columns}")
    df_cov = df_cov[selected_columns]  # Select only the specified columns
else:
    logger.info("Using all columns from the covariates file")

columns_string = '-'.join(list(df_cov.columns))
unique_id += '-' + columns_string

Y = result_matrix.to_numpy()
Y = torch.Tensor(Y)

c = torch.Tensor([pd.factorize(df_cov[c])[0] for c in df_cov.columns]).T

logger.info(f"Covariates shape: {c.shape} ({df_cov.columns})")
logger.info(f"Features shape: {Y.shape}")

data_dim = Y.shape[1]
n_covariates = c.shape[1]
hidden_dim = 128
z_dim = 1
lim_val = 2.0
steps = 15
# grid needed for quadrature
grid_z = torch.linspace(-lim_val, lim_val, steps=steps).reshape(-1, 1).to(device)
grid_cov = torch.linspace(-lim_val, lim_val, steps=steps).reshape(-1, 1).to(device)
grid_c = [grid_cov for _ in range(n_covariates)]

dataset = TensorDataset(Y.to(device), c.to(device))
data_loader = DataLoader(dataset, shuffle=True, batch_size=16)

encoder_mapping = nn.Sequential(
    nn.Linear(data_dim + n_covariates, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 2*z_dim)
)

decoder_z = nn.Sequential(
    nn.Linear(z_dim, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, data_dim)
)

decoder_c = nn.Sequential(
    nn.Linear(1, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, data_dim)
)

decoder_cz = nn.Sequential(
    nn.Linear(1 + z_dim, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, data_dim)
)

encoder = cEncoder(z_dim=z_dim, mapping=encoder_mapping)

decoders_c = [decoder_c for _ in range(n_covariates)]
decoders_cz = [decoder_cz for _ in range(n_covariates)]

decoder = Decoder_multiple_covariates(data_dim, n_covariates,
                  grid_z, grid_c, 
                  decoder_z, decoders_c, decoders_cz,
                  has_feature_level_sparsity=True, p1=0.1, p2=0.1, p3=0.1, 
                  lambda0=1e3, penalty_type="MDMM",
                  device=device)

model = CVAE_multiple_covariates(encoder, decoder, lr=5e-4, device=device)

logger.info(f"Started training")
loss, integrals = model.optimize(data_loader,
                                 n_iter=n_iter, 
                                 augmented_lagrangian_lr=0.1)
logger.info(f"Finished training")

below_02 = len((abs(integrals[:, -1]) < 0.2).nonzero()[0]) / len(integrals)
below_01 = len((abs(integrals[:, -1]) < 0.1).nonzero()[0]) / len(integrals)

logger.info(f"Computed {len(integrals)} integrals. {below_02:.2f} below 0.2, {below_01:.2f} below 0.1")

plot_integrals(integrals[:100], filename=f"results/{unique_id}_integrals_plot.png")

with torch.no_grad():
    mu_z, sigma_z = encoder(Y.to(device), c.to(device))
    
    Y_pred = decoder(mu_z, c.to(device)).cpu()
    Y_error = Y - Y_pred
    Y_error = Y_error.cpu()

    mu_z, sigma_z = mu_z.cpu(), sigma_z.cpu()

    min_z, max_z = mu_z.min(), mu_z.max()
    num_points = 100
    z_linear_space = torch.linspace(min_z, max_z, num_points)
    z_linear_space = z_linear_space.view(-1, 1).to(device)

    Y_preds = {}
    labels = {}

    Y_preds['Z'] = {}
    Y_preds['Z'][0] = decoder_z(z_linear_space).cpu()
    labels['Z'] = {'values': [0]*mu_z.shape[0], 'names': {0: ''}}

    for c_val in range(n_covariates):
        results_cof = {}
        results_int = {}
        column = df_cov[df_cov.columns[c_val]]
        logger.info(f"Unique values for covariate {df_cov.columns[c_val]}: {len(column.unique())}")
        uniques = torch.unique(c[:, c_val])
        for u in uniques:
            col1 = torch.full((100, 1), u).float()  # Column of cofactors values
            results_cof[int(u)] = decoders_c[c_val](col1.to(device)).cpu()
            results_int[int(u)] = decoders_cz[c_val](torch.cat([z_linear_space, col1.to(device)], dim=1)).cpu()

        column_name = df_cov.columns[c_val]

        Y_preds[f'{column_name}'] = results_cof
        Y_preds[f'Z-{column_name}'] = results_int

        labels[f'{column_name}'] = {'values': c[:, c_val], 'names': {int(k): v for k, v in enumerate(pd.factorize(df_cov[column_name])[1])}}
        labels[f'Z-{column_name}'] = labels[f'{column_name}']

varexp = decoder.fraction_of_variance_explained(mu_z.to(device), c.to(device), Y_error=Y_error.to(device)).cpu()

column_names = ['Z'] + list(df_cov.columns) + [f'Z-{c}' for c in list(df_cov.columns)] + ['Noise']

column_order = ['Z']
cov_indices = {'Z': 0}
for i, col in enumerate(df_cov.columns):
    column_order.append(col)
    cov_indices[col] = i
    column_order.append(f'Z-{col}')
    cov_indices[f'Z-{col}'] = i
column_order.append('Noise')

plot_variance(varexp, column_names, column_order, filename=f"results/{unique_id}_variance_plot.png")

indices = []
for i in range(varexp.shape[1] - 1):
    index = torch.argmax(varexp[:, i]).item()
    logger.info(f"Most explained by {column_names[i]} - {result_matrix.columns[index]}")
    indices.append(index)

plot_features(
    indices,
    z_linear_space.cpu(),
    mu_z,
    Y,
    Y_pred,
    Y_preds,
    varexp,
    result_matrix,
    column_order,
    labels,
    c,
    cov_indices,
    f"results/{unique_id}_feature_plot.png",
)

logger.info(f"Finish")