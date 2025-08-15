# fed_server.py
"""
1. Learning rate
    client_configs = {
    "0": {"local_epochs": 5, "batch_size": 256, "lr": 1e-3},
    "1": {"local_epochs": 1, "batch_size": 64,  "lr": 5e-4},
    # …
}
And in the client’s fit(), reset the optimizer:
lr = config.get("lr", 1e-3)
self.optimizer = Adam(self.model.parameters(), lr=lr)
---------------------------------------
2. Optimizer type
client_configs["2"] = {"optimizer": "SGD", "lr": 1e-2}
# in fit():
opt_name = config.get("optimizer", "Adam")
self.optimizer = {"Adam": Adam, "SGD": torch.optim.SGD}[opt_name](
    self.model.parameters(), lr=config["lr"]
)
-----------------------------------
3. Model personalization / architecture

client_configs["3"] = {"tf_layers": 1, "tab_depth": 2}
# in client init:
layers = config.get("tf_layers", 2)
depth  = config.get("tab_depth", 6)
self.model = CombinedRULModel(..., layers=layers, depth=depth)
-----------------------------------
4. Data augmentation / preprocessing
client_configs["0"]["augment"] = True
# in fit():
if config.get("augment"):
    x_ts = augment_fn(x_ts)
----------------------------------
5. Gradient clipping or DP settings
client_configs["1"]["clip_norm"] = 1.0
# in fit():
torch.nn.utils.clip_grad_norm_(self.model.parameters(), config["clip_norm"])
------------------------------------


6. Communication compression / quantization
Feed clients a compress: True flag, then apply weight quantization before sending updates.
------------------------------------
7. Local evaluation frequency
Tell some clients to run an extra local validation pass every N epochs by passing eval_every: 5, for instance.
------------------
8.Subset‐of‐layers training
client_configs["2"]["freeze_backbone"] = True
# in fit():
if config.get("freeze_backbone"):
    for p in self.model.tf.parameters(): p.requires_grad = False

"""

import os
import torch
import flwr as fl
from typing import Dict, Optional, Tuple, List







# === Custom Strategy for Heterogeneous Clients ===
class HeterogeneousStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        save_path: str,
        client_configs: Dict[str, Dict],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_path = save_path
        self.client_configs = client_configs
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        # Sample all available clients
        clients = client_manager.all().values()
        fit_ins_list = []
        for client in clients:
            # Identify clients by their CID
            cid = client.cid
            # Fetch per-client config or default
            cfg = self.client_configs.get(cid, {})
            # Always include global parameters
            fit_ins = fl.common.FitIns(parameters, cfg)
            fit_ins_list.append((client, fit_ins))
        return fit_ins_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures,
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        aggregated_parameters, agg_metrics = super().aggregate_fit(server_round, results, failures)
        # Save global model
        if aggregated_parameters is not None:
            weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            path = self.save_path.format(round=server_round)
            torch.save(weights, path)
            print(f"🔖 Saved global model for round {server_round} → {path}")
        return aggregated_parameters, agg_metrics

# === Server Entry Point ===
def main():

    NUM_CLIENTS = 2
    # Path to save models
    SAVE_PATH = "./artifacts/global_model_round_{round}.pt"
    # Define different client capabilities
    # e.g. client '0' is powerful (more epochs), '1' is weak (fewer epochs)
    client_configs = {
        # "0": {"local_epochs": 5, "batch_size": 256},
        # "1": {"local_epochs": 1, "batch_size": 64}
        # Clients '2' and '3' use default
    
    # "0": {"local_epochs": 5, "batch_size": 256, "lr": 1e-3},
    # "1": {"local_epochs": 1, "batch_size": 64,  "lr": 5e-4}
    
    
    # "0": {"local_epochs": 5, "batch_size": 256, "use_dp": True, "dp_sigma": 0.1, "dp_clip_bound": 1.0, "dp_spec_k": 2},
    # "1": {"local_epochs": 5, "batch_size": 256, "use_dp": False}

    
      "0": {"local_epochs": 5, "batch_size": 256, "use_dp": True, "dp_sigma": 0.1, "dp_clip_bound": 1.0, "dp_spec_k": 2},
    "1": {"local_epochs": 5, "batch_size": 256, "use_dp": True, "dp_sigma": 0.1, "dp_clip_bound": 1.0, "dp_spec_k": 2}
    
    
    
    
    
    }




    strategy = HeterogeneousStrategy(
        save_path=SAVE_PATH,
        client_configs=client_configs,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=lambda metrics: {
            "mse": (
                sum(num * m.get("mse", 0.0) for num, m in metrics)
                / max(1, sum(num for num, _ in metrics))
            )
        },
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()







"""
#fl.server.strategy.FedProx from flwr.server.strategy import FedProx
Adds a proximal term to each client’s objective to limit how far local weights can drift from the global model. Great for heterogeneous data.
strategy = FedProx(
    mu=0.01,                              # proximal regularization strength
    fraction_fit=0.5,                     # sample 50% clients per round
    fraction_evaluate=0.5,
    min_fit_clients=NUM_CLIENTS//2,
    min_evaluate_clients=NUM_CLIENTS//2,
    min_available_clients=NUM_CLIENTS,
)

#FedAdam, FedYogi, FedAdagrad  hese mirror their namesake optimizers on the server side, applying adaptive moment updates to the global model. They can improve convergence when client updates are noisy.
from flwr.server.strategy import FedAdam

strategy = FedAdam(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    server_learning_rate=1e-2,    # server‐side adaptivity
)


QFedAvg (flwr.server.strategy.QFedAvg)
A “fairness‐aware” variant that reduces the influence of clients with extremely large losses, giving a more equitable global model across all participants.
from flwr.server.strategy import QFedAvg

strategy = QFedAvg(
    q=0.5,                         # fairness‐tradeoff parameter
    fraction_fit=0.8,
    min_fit_clients=NUM_CLIENTS//2,
    min_available_clients=NUM_CLIENTS,
)


AdaptiveFedAvg
from flwr.server.strategy import AdaptiveFedAvg
A variant that adapts the learning rate of the global model based on client performance, improving convergence

"""








"""
1. Learning rate
    client_configs = {
    "0": {"local_epochs": 5, "batch_size": 256, "lr": 1e-3},
    "1": {"local_epochs": 1, "batch_size": 64,  "lr": 5e-4},
    # …
}
And in the client’s fit(), reset the optimizer:
lr = config.get("lr", 1e-3)
self.optimizer = Adam(self.model.parameters(), lr=lr)
---------------------------------------
2. Optimizer type
client_configs["2"] = {"optimizer": "SGD", "lr": 1e-2}
# in fit():
opt_name = config.get("optimizer", "Adam")
self.optimizer = {"Adam": Adam, "SGD": torch.optim.SGD}[opt_name](
    self.model.parameters(), lr=config["lr"]
)
-----------------------------------
3. Model personalization / architecture

client_configs["3"] = {"tf_layers": 1, "tab_depth": 2}
# in client init:
layers = config.get("tf_layers", 2)
depth  = config.get("tab_depth", 6)
self.model = CombinedRULModel(..., layers=layers, depth=depth)
-----------------------------------
4. Data augmentation / preprocessing
client_configs["0"]["augment"] = True
# in fit():
if config.get("augment"):
    x_ts = augment_fn(x_ts)
----------------------------------
5. Gradient clipping or DP settings
client_configs["1"]["clip_norm"] = 1.0
# in fit():
torch.nn.utils.clip_grad_norm_(self.model.parameters(), config["clip_norm"])
------------------------------------


6. Communication compression / quantization
Feed clients a compress: True flag, then apply weight quantization before sending updates.
------------------------------------
7. Local evaluation frequency
Tell some clients to run an extra local validation pass every N epochs by passing eval_every: 5, for instance.
------------------
8.Subset‐of‐layers training
client_configs["2"]["freeze_backbone"] = True
# in fit():
if config.get("freeze_backbone"):
    for p in self.model.tf.parameters(): p.requires_grad = False

"""