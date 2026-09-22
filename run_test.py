import pandas as pd
import ollama

metrics = pd.read_parquet("RCAEval-data/re1ob_adservice_cpu_1/metrics.parquet")
idx = pd.read_parquet("RCAEval-data/cases.parquet")
ground_truth = idx[idx.case == "re1ob_adservice_cpu_1"]["root_cause_service"].values[0]

# Split into "normal" (first half) vs "faulty" (second half) periods
mid = len(metrics) // 2
normal = metrics.iloc[:mid]
faulty = metrics.iloc[mid:]

# Compute how much each column's mean shifted, normalized by its normal-period std dev
anomaly_scores = {}
for col in metrics.columns:
    if col == "time":
        continue
    std = normal[col].std()
    if std == 0 or pd.isna(std):
        continue
    shift = abs(faulty[col].mean() - normal[col].mean()) / std
    anomaly_scores[col] = shift

# Top 10 most anomalous columns
top = sorted(anomaly_scores.items(), key=lambda x: -x[1])[:10]
summary = "\n".join(f"{col}: anomaly score {score:.2f}" for col, score in top)

prompt = f"""You are a root cause analysis system for a microservices application.

Below are the top 10 metrics with the largest statistical shift between the normal period and the faulty period (higher score = more anomalous, measured in standard deviations).

{summary}

Column names follow the pattern service_metric. Identify which ONE service is most likely the root cause.

Respond with ONLY the service name, nothing else."""

#response = ollama.chat(model="qwen2.5-coder:7b", messages=[{"role": "user", "content": prompt}])
response = ollama.chat(model="gemma4:26b", messages=[{"role": "user", "content": prompt}])
answer = response["message"]["content"].strip()

print("Top anomalies:\n", summary)
print("\nModel answered:", answer)
print("Ground truth was:", ground_truth)
print("Correct!" if ground_truth.lower() in answer.lower() else "Incorrect.")