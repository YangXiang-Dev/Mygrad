import json
import random
import numpy as np
from sklearn.datasets import make_moons
from engine import Value, MLP

random.seed(1337)
np.random.seed(1337)

X, y = make_moons(100, noise=0.1)
model = MLP(2, [10, 10, 1])

h = 0.075
x_min, x_max = float(X[:, 0].min() - 1), float(X[:, 0].max() + 1)
y_min, y_max = float(X[:, 1].min() - 1), float(X[:, 1].max() + 1)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]

print(f"Grid: {xx.shape[0]}x{xx.shape[1]} = {len(grid_points)} points")

def compute_boundary():
    return [1 if model(list(map(Value, row))).data > 0.5 else 0 for row in grid_points]

def loss_fn():
    scores = [model(a) for a in X]
    loss_result = sum((s - b) ** 2 for s, b in zip(scores, y)) * (1.0 / len(X))
    acc = sum((s.data > 0.5) == (b > 0.5) for s, b in zip(scores, y)) / len(X)
    return loss_result, acc

lr = 0.05
epochs = 300
steps = []
boundaries = {}

for k in range(epochs):
    model.clear_grad()
    loss_result, acc = loss_fn()
    loss_result.backward()
    for p in model.parameters():
        p.data -= lr * p.grad

    steps.append({"step": k, "loss": float(loss_result.data), "acc": float(acc)})
    boundaries[k] = compute_boundary()
    print(f"step {k} loss {loss_result.data:.4f}, acc {acc:.2%}")

data = {
    "X": X.tolist(),
    "y": y.tolist(),
    "grid": {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "h": h,
        "rows": int(xx.shape[0]),
        "cols": int(xx.shape[1]),
    },
    "steps": steps,
    "boundaries": {str(k): v for k, v in boundaries.items()},
}

with open("D:/projects/GPTLearn/mygrad/training_data.json", "w") as f:
    json.dump(data, f)

print(f"\nExported to training_data.json ({len(steps)} steps, {len(boundaries)} boundaries)")
