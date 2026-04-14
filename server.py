# type: ignore
import flwr as fl

# =====================
# 🔥 METRICS AGGREGATION (VERY IMPORTANT)
# =====================
def weighted_average(metrics):
    if len(metrics) == 0:
        return {}

    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

# =====================
# STRATEGY
# =====================
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,

    # 🔥 IMPORTANT FIX
    evaluate_metrics_aggregation_fn=weighted_average,
)

# =====================
# MAIN
# =====================
def main():
    print("🚀 Federated Server Start ho raha hai...")

    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

# =====================
# RUN
# =====================
if __name__ == "__main__":
    main()