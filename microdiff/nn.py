import random

class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [random.uniform(-1,1) for _ in range(nin)]
        self.b = 0
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum([wi*xi for wi,xi in zip(self.w, x)], self.b)
        return ReLU(act) if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'} Neuron({len(self.w)})"

def ReLU(x):
    return max(0, x)

if __name__ == "__main__":
    pass