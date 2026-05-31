# FederatedApproach — Federated Learning with Flower

This folder implements federated training of the `CombinedRULModel` (TimeSeriesEmbedder + TabTransformer) using the [Flower (`flwr`)](https://flower.dev/) library. Two federation strategies are supported: heterogeneous and homogeneous clients.

---

## Scripts

### FedDiff — Heterogeneous Federation

#### `feddiff_server.py`

Custom `HeterogeneousStrategy` extending `FedAvg`. Key behaviour:
- **Per-client config dispatch**: each client gets its own `config` dict (local epochs, batch size, LR, DP settings) before every training round.
- **Model saving**: after every aggregation, the global model weights are saved to `artifacts/global_model_round_{round}.pt`.
- **Configurable client capabilities** via `client_configs` dict — examples include:
  - Different local epochs (simulating compute power differences)
  - Different batch sizes (simulating memory constraints)
  - Different LRs (simulating dataset scale differences)
  - Per-client DP on/off with Spectral-DP parameters

#### `feddiff_client.py`

Base heterogeneous `flwr.client.NumPyClient`. Reads per-round config in `fit()` and adapts:
- Rebuilds DataLoader with the configured `batch_size`
- Resets optimizer with configured `lr`
- Data is partitioned by vehicle ID (non-overlapping fleets across clients)

#### `feddiff_client+spectraldp.py`

Extends `feddiff_client.py` with **Spectral-DP** gradient perturbation. When the server sends `use_dp=True`, after each backward pass the client runs `spectral_dp_gradient_update(model, sigma, clip_bound, spec_k)` before `optimizer.step()`.

### FedSame — Homogeneous Federation

#### `fedsame_server.py`

Standard `FedAvg` server with optional model checkpointing. All clients receive the same config.

#### `fedsame_client.py`

Homogeneous client. All clients use identical training hyperparameters. Suited for benchmarking or uniform deployments.

---

## Client Configuration Options (FedDiff)

| Config Key | Description | Example |
|---|---|---|
| `local_epochs` | Local training epochs per round | 1–5 |
| `batch_size` | DataLoader batch size | 64–256 |
| `lr` | Learning rate for Adam optimizer | 5e-4 – 1e-3 |
| `use_dp` | Enable Spectral-DP on this client | `True`/`False` |
| `dp_sigma` | Noise multiplier σ for Spectral-DP | 0.1 |
| `dp_clip_bound` | Singular value clip bound C | 1.0 |
| `dp_spec_k` | Top-k singular values to retain (None = all) | 2 |

---

## Data Partitioning

Each client receives a disjoint subset of vehicles:
```python
unique = np.unique(vids)
splits = np.array_split(unique, num_clients)
mask = np.isin(vids, splits[client_id])
```
This simulates separate industrial fleets that never share raw data.

---

## Running

```bash
# Terminal 1 — start server
python FederatedApproach/feddiff_server.py

# Terminal 2 — client 0 (with Spectral-DP)
python FederatedApproach/feddiff_client+spectraldp.py --client-id 0

# Terminal 3 — client 1
python FederatedApproach/feddiff_client+spectraldp.py --client-id 1
```

---

## Results

<img width="692" height="320" alt="Client training output" src="https://github.com/user-attachments/assets/1608d28c-70eb-412d-8a5e-6899d87f79cb" />
<img width="708" height="166" alt="Server round log" src="https://github.com/user-attachments/assets/52bb5040-415f-49bb-800e-49a6c9fda410" />
<img width="673" height="716" alt="Heterogeneous client results" src="https://github.com/user-attachments/assets/32209bdd-0f91-4ca1-b7df-8cc1314b1576" />
<img width="524" height="319" alt="Federated convergence" src="https://github.com/user-attachments/assets/61c75505-b536-49ce-b393-4ac4d2bfacff" />
