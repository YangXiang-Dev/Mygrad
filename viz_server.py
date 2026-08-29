import json
import threading
import random
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from sklearn.datasets import make_moons
from engine import Value, MLP
import os
import webbrowser

random.seed(1337)
np.random.seed(1337)

X, y = make_moons(100, noise=0.1)
model = MLP(2, [10, 10, 1])

h = 0.075
x_min, x_max = float(X[:, 0].min() - 1), float(X[:, 0].max() + 1)
y_min, y_max = float(X[:, 1].min() - 1), float(X[:, 1].max() + 1)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]

steps_meta = []
boundaries = {}

static_info = {
    "X": X.tolist(),
    "y": y.tolist(),
    "grid": {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "h": h,
        "rows": int(xx.shape[0]),
        "cols": int(xx.shape[1]),
    },
}

done = False

def compute_boundary():
    return [1 if model(list(map(Value, row))).data > 0.5 else 0 for row in grid_points]

def loss_fn():
    scores = [model(a) for a in X]
    loss_result = sum((s - b) ** 2 for s, b in zip(scores, y)) * (1.0 / len(X))
    acc = sum((s.data > 0.5) == (b > 0.5) for s, b in zip(scores, y)) / len(X)
    return loss_result, acc

def train():
    global done
    lr = 0.05
    epochs = 300

    for k in range(epochs):
        model.clear_grad()
        loss_result, acc = loss_fn()
        loss_result.backward()
        for p in model.parameters():
            p.data -= lr * p.grad

        steps_meta.append({"step": k, "loss": float(loss_result.data), "acc": float(acc)})
        boundaries[k] = compute_boundary()
        print(f"step {k} loss {loss_result.data:.4f}, acc {acc:.2%}")

    done = True
    print("Training done!")

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/meta":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                **static_info,
                "steps": steps_meta,
                "done": done,
            }).encode())
        elif self.path.startswith("/api/boundary/"):
            step = int(self.path.split("/")[-1])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            b = boundaries.get(step, [])
            self.wfile.write(json.dumps(b).encode())
        elif self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open(os.path.join(os.path.dirname(__file__), "viz.html"), "rb") as f:
                self.wfile.write(f.read())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        print(f"[HTTP] {args[0] if args else ''}")

class ThreadedServer(ThreadingMixIn, HTTPServer):
    pass

PORT = 8085
threading.Thread(target=train, daemon=True).start()
print(f"Open http://localhost:{PORT}")
webbrowser.open(f"http://localhost:{PORT}")
ThreadedServer(("", PORT), Handler).serve_forever()
