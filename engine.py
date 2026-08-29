class Value:
    def __init__(self, data, _backward=lambda:None, _prev=()):
        self.data = data
        self.grad = 0
        self._backward = _backward
        self._prev = _prev

    def backward(self):
        visited = set()

        result = []

        def topo(node):
            if node not in visited:
                visited.add(node)
                for n in node._prev:
                    topo(n)
                result.append(node)

        topo(self)

        self.grad = 1
        backward_list = reversed(list(result))
        for n in backward_list: n._backward()

    def __add__(self, other):
        if not isinstance(other,Value):
            other = Value(other)
        out = Value(self.data + other.data)
        def out_back():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = out_back
        out._prev = [self, other]
        return out 

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other,Value):
            other = Value(other)
        out = Value(self.data * other.data)
        def out_back():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = out_back
        out._prev = [self, other]
        return out 

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + -other

    def __neg__(self):
        return self*(-1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int/float"
            
        out = Value(self.data ** other)
        def out_back():
            self.grad += out.grad * self.data ** (other-1) * other
        out._backward = out_back
        out._prev = [self]
        return out 

    def __truediv__(self, other):
        return self * (other ** -1)

    def relu(self):
        out = Value(self.data if self.data > 0 else 0)
        def out_back():
            self.grad += out.grad if out.data > 0 else 0
        out._backward = out_back
        out._prev = [self]
        return out 



class Module:
    def parameters(self):
        return []

    def clear_grad(self):
        for p in self.parameters():
            p.grad = 0

import random

class Neuron(Module):
    def __init__(self, nin,activate=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self.activate = activate

    def __call__(self,x):
        res = sum(a*b for (a,b) in zip(self.w,x)) + self.b
        if(self.activate):
            res = res.relu()
        return res

    def parameters(self):
        return self.w + [self.b]
        

class Layer(Module):
    def __init__(self,nin, nout, **kwargs):
        self.nodes = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self,x):
        res = [n(x) for n in self.nodes]
        if len(res) == 1:
            res = res[0]
        return res
    
    def parameters(self):
        return [p for node in self.nodes for p in node.parameters()]

class MLP(Module):
    def __init__(self, nin, nouts):
        num_list = [nin] + nouts
        self.layers = [Layer(num_list[k], num_list[k+1], activate=True if k != len(num_list)-2 else False) for k in range(len(num_list)-1)]

    def __call__(self, x):
        res = x
        for layer in self.layers:
            res = layer(res)
        return res

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



if __name__ == "__main__":
    model = MLP(2, [10,10,1])

    import numpy as np
    from sklearn.datasets import make_moons

    X, y = make_moons(100, noise=0.1)

    def loss(batch_size=None):
        if batch_size==None:
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

    for k in range(300):

        model.clear_grad()

        loss_result,acc = loss(16)
        loss_result.backward()

        for p in model.parameters():
            p.data -= lr*p.grad

        print(f"step {k} loss {loss_result.data:.4f}, acc {acc:.2%}")