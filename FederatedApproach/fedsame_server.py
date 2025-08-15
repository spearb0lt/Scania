# fed_server.py
# import flwr as fl

# def main():
#     strategy = fl.server.strategy.FedAvg(
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_fit_clients=2,
#         min_evaluate_clients=2,
#         min_available_clients=2,
#     )
#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=2),
#         strategy=strategy,
#     )

# if __name__ == "__main__":
#     main()


# fed_server.py
import os
import torch
import flwr as fl
from typing import Dict, Optional, Tuple, List

# === Custom Strategy to Save Global Model ===


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        save_path: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures,
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        # Perform standard aggregation
        aggregated_parameters, agg_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            # Convert Parameters to tensor dict
            weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Save global model weights
            torch.save(weights, self.save_path.format(round=server_round))
            print(f"🔖 Saved global model for round {server_round}→ {self.save_path.format(round=server_round)}")
        return aggregated_parameters, agg_metrics

# === Server Implementation ===
def main():
    # Path template for saving per-round global models
    SAVE_PATH = "./artifacts/global_model_round_{round}.pt"
    num_clients = 2
    # Instantiate custom strategy
    strategy = SaveModelStrategy(
        save_path=SAVE_PATH,
        fraction_fit=1.0, #100% of the available clients are asked to participate in training every round
        fraction_evaluate=1.0,#100% of the available clients are asked to run evaluation every round.
        min_fit_clients=num_clients,#You require at least this many clients to successfully complete a fit() call before the server will proceed with aggregation.
        min_evaluate_clients=num_clients,#You require at least this many clients to successfully run evaluate() before the server records the round’s metrics.
        min_available_clients=num_clients, #The server will not start until at least this many clients are available.
        # Weighted MSE aggregation
        fit_metrics_aggregation_fn=lambda metrics: {
            "mse": (
                sum(num * m.get("mse", 0.0) for num, m in metrics)
                / max(1, sum(num for num, _ in metrics))
            )
        },
    )

    # Start Flower server (blocks until done)
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    print("✅ Federated training complete. Global models saved at:")
    print(SAVE_PATH.format(round="*"))

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