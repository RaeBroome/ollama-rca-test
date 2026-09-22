import pandas as pd
import ollama

CASE = "re3ss_carts_f1_1"
MODEL = "qwen2.5-coder:7b"
INJECT_TIME = 1732243203

logs = pd.read_parquet(f"RCAEval-data/{CASE}/logs.parquet")
idx = pd.read_parquet("RCAEval-data/cases.parquet")
ground_truth = idx[idx.case == CASE]["root_cause_service"].values[0]

pd.set_option('display.max_colwidth', None)

window = logs[(logs.timestamp >= INJECT_TIME - 15) & (logs.timestamp <= INJECT_TIME + 60)]

# Only the two services we confirmed by hand contain the actual signal
signal_logs = window[window.container_name.isin(["carts", "carts-db"])]

log_text = signal_logs.to_string(index=False)
print(f"Signal lines: {len(signal_logs)}\n")
print(log_text[:500], "...\n")

prompt = f"""You are a root cause analysis system for a microservices application called Sock Shop.

Below are log lines from two related services: carts (the cart application) and carts-db (its database).

Logs:
{log_text}

Which ONE service is the true root cause of the failure — carts or carts-db? Respond with ONLY the service name."""

response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
answer = response["message"]["content"].strip()

print("Model:", MODEL)
print("Model answered:", answer)
print("Ground truth was:", ground_truth)
print("Correct!" if answer.lower().strip() == ground_truth.lower() else "Incorrect.")