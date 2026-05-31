# FederatedApproach -- Federated Learning with Flower

This folder implements federated training of the `CombinedRULModel` (TimeSeriesEmbedder
+ TabTransformer) using the Flower (flwr) library. Two federation strategies are
supported: heterogeneous (FedDiff) and homogeneous (FedSame).

---

## Scripts

### FedDiff -- Heterogeneous Federation

#### `feddiff_server.py`

Custom `HeterogeneousStrategy` extending `FedAvg`. Key behaviour:
- **Per-client config dispatch**: before each round, dispatches a different `config`
  dict to each client (local epochs, batch size, LR, DP settings).
- **Model saving**: after every aggregation round, saves global weights to
  `artifacts/global_model_round_{round}.pt`.
- Configurable via `client_configs` dict at top of `main()`.

#### `feddiff_client.py`

Base heterogeneous `flwr.client.NumPyClient`. Reads per-round config in `fit()`:
- Rebuilds `DataLoader` with the configured `batch_size`
- Resets optimizer with configured `lr`
- Trains for configured `local_epochs`
- Data partitioned by vehicle ID (disjoint fleets per client)

#### `feddiff_client+spectraldp.py`

Extends `feddiff_client.py` with Spectral-DP support. When the server sends
`use_dp=True`, after each backward pass the client applies
`spectral_dp_gradient_update(model, sigma, clip_bound, spec_k)` before
`optimizer.step()`.

### FedSame -- Homogeneous Federation

#### `fedsame_server.py`

Standard `FedAvg` server with optional model checkpointing. All clients receive
the same training configuration. For benchmarking or uniform deployments.

#### `fedsame_client.py`

Homogeneous client -- identical training hyperparameters for every client in
every round. Suited for establishing baseline federated performance.

---

## Per-Client Configuration (FedDiff)

| Key | Type | Description | Example |
|---|---|---|---|
| `local_epochs` | int | Training epochs per round | 1-5 |
| `batch_size` | int | DataLoader batch size | 64-256 |
| `lr` | float | Adam learning rate | 5e-4 to 1e-3 |
| `use_dp` | bool | Enable Spectral-DP | True / False |
| `dp_sigma` | float | Spectral-DP noise multiplier sigma | 0.1 |
| `dp_clip_bound` | float | Singular value clip bound C | 1.0 |
| `dp_spec_k` | int or None | Top-k singular values to keep | 2 or None |

Additional extensible parameters (documented in server file):
- `optimizer`: "Adam" or "SGD"
- `freeze_backbone`: freeze TimeSeriesEmbedder weights
- `augment`: apply data augmentation
- `clip_norm`: gradient norm clipping
- `eval_every`: local validation frequency

---

## Data Partitioning

Each client receives a disjoint subset of vehicles -- simulating completely
separate industrial fleets:

```python
unique_vehicles = np.unique(window_vids)
splits = np.array_split(unique_vehicles, num_clients)
mask = np.isin(window_vids, splits[client_id])
X_client, y_client, specs_client = X[mask], y[mask], specs[mask]
```

No raw sensor data is ever shared across federation boundaries.

---

## Running

```bash
# Terminal 1 -- server
python FederatedApproach/feddiff_server.py

# Terminal 2 -- client 0 (powerful, Spectral-DP enabled)
python FederatedApproach/feddiff_client+spectraldp.py --client-id 0

# Terminal 3 -- client 1 (weaker, no DP or lighter config)
python FederatedApproach/feddiff_client+spectraldp.py --client-id 1
```

---

## Results

<img width="692" height="320" alt="Client training output" src="https://github.com/user-attachments/assets/1608d28c-70eb-412d-8a5e-6899d87f79cb" />
<img width="708" height="166" alt="Server round log" src="https://github.com/user-attachments/assets/52bb5040-415f-49bb-800e-49a6c9fda410" />
<img width="673" height="716" alt="Heterogeneous client results" src="https://github.com/user-attachments/assets/32209bdd-0f91-4ca1-b7df-8cc1314b1576" />
<img width="524" height="319" alt="Federated convergence" src="https://github.com/user-attachments/assets/61c75505-b536-49ce-b393-4ac4d2bfacff" />
