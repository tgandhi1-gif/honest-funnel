"""The Honest Funnel - last click versus Markov removal effect attribution."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)
channels = ["Search", "Social", "Email", "Display", "Referral"]
true_power = np.array([0.34, 0.14, 0.28, 0.06, 0.18])

N = 40000
paths = []
converted = []
for _ in range(N):
    length = rng.integers(1, 5)
    path = list(rng.choice(len(channels), size=length, p=true_power / true_power.sum()))
    score = true_power[path].sum() / 2
    conv = rng.random() < min(score, 0.9)
    paths.append(path)
    converted.append(conv)

converted = np.array(converted)

last_click = np.zeros(len(channels))
for path, c in zip(paths, converted):
    if c:
        last_click[path[-1]] += 1
last_click = last_click / last_click.sum()

# removal effect: drop each channel, see how conversions fall
base = converted.sum()
removal = np.zeros(len(channels))
for ci in range(len(channels)):
    kept = [c for path, c in zip(paths, converted) if ci not in path]
    removal[ci] = base - sum(kept)
markov = removal / removal.sum()

df = pd.DataFrame({"last_click": last_click, "markov_truth": markov}, index=channels)
print(df.round(3))
shift = np.abs(df["last_click"] - df["markov_truth"]).sum() / 2
print(f"Spend reallocated once attribution is honest: {shift*100:.0f}%")

os.makedirs("outputs", exist_ok=True)
df.plot(kind="bar", color=["#cccccc", "#ff6a3d"], figsize=(9, 5))
plt.ylabel("share of credit")
plt.title("Last click versus honest (Markov) attribution")
plt.tight_layout()
plt.savefig("outputs/honest_funnel.png", dpi=120)
print("Saved outputs/honest_funnel.png")
