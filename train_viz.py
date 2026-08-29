import random
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.datasets import make_moons
from engine import Value, MLP

random.seed(1337)
np.random.seed(1337)

X, y = make_moons(100, noise=0.1)

model = MLP(2, [10, 10, 1])

checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
fig_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(fig_dir, exist_ok=True)

def save_checkpoint(model, step, loss_val, acc_val):
    data = {
        "step": step,
        "loss": loss_val,
        "acc": acc_val,
        "params": [p.data for p in model.parameters()]
    }
    path = os.path.join(checkpoint_dir, f"step_{step:04d}.json")
    with open(path, "w") as f:
        json.dump(data, f)

def load_checkpoint(model, path):
    with open(path, "r") as f:
        data = json.load(f)
    for p, val in zip(model.parameters(), data["params"]):
        p.data = val
    return data["step"]

def plot_decision_boundary(model, X, y, step, loss_val, acc_val):
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    scores = [model(list(map(Value, xrow))).data for xrow in Xmesh]
    Z = np.array([s > 0.5 for s in scores]).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors="k")
    plt.title(f"Step {step} | Loss {loss_val:.4f} | Acc {acc_val:.0%}")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig(os.path.join(fig_dir, f"step_{step:04d}.png"), dpi=100, bbox_inches="tight")
    plt.close()

def loss(batch_size=None):
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]

    def loss_fun(y_true, y_test):
        return (y_test - y_true) ** 2

    scores = [model(a) for a in Xb]
    loss_result = sum(loss_fun(b, s) for (b, s) in zip(yb, scores)) * (1.0 / len(Xb))
    acc = sum((s.data > 0.5) == (b > 0.5) for s, b in zip(scores, yb)) / len(Xb)
    return loss_result, acc

lr = 0.05
epochs = 300
viz_every = 20

history = {"loss": [], "acc": []}

for k in range(epochs):
    model.clear_grad()
    loss_result, acc = loss()
    loss_result.backward()

    for p in model.parameters():
        p.data -= lr * p.grad

    history["loss"].append(loss_result.data)
    history["acc"].append(acc)

    print(f"step {k} loss {loss_result.data:.4f}, acc {acc:.2%}")

    if k % viz_every == 0 or k == epochs - 1:
        plot_decision_boundary(model, X, y, k, loss_result.data, acc)
        save_checkpoint(model, k, loss_result.data, acc)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history["loss"])
plt.title("Loss")
plt.xlabel("Step")
plt.subplot(1, 2, 2)
plt.plot(history["acc"])
plt.title("Accuracy")
plt.xlabel("Step")
plt.savefig(os.path.join(fig_dir, "training_curves.png"), dpi=100, bbox_inches="tight")
plt.close()

print(f"\nDone. Figures saved to {fig_dir}, checkpoints saved to {checkpoint_dir}")
