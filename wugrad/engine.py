class Value:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0  # we assume that at initialization, every value has NO effect on the function.

        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc
        self.label = label

    def __repr__(self):
        if self.label != "":
            return f"Value(data={self.data}, label='{self.label}')"
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        # propagate gradient to 'children' / individual expression terms
        # out = self + other
        # dL / d_self = dL / d_out * d_out / d_self = out.grad * d(self + other)/d_self = out.grad * 1.0
        # dL / d_other = dL / d_out * d_out / d_other = out.grad * d(self + other)/d_other = out.grad * 1.0
        def _backward():
            # we use += because of the multivariate chain rule
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        # f(out(self,other)) = self * other
        # out = self * other.
        # -> f'(out) * out'(self) = out.grad * other
        def _backward():
            # we need to do += because of the Multivariable Chain Rule
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, other):  # self^other
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        # f(out(self,other)) = self ** other
        # out = self ** other.
        # -> df/ dself = f'(out) * out'(self) = out.grad * (other * self ** other-1)
        def _backward():
            self.grad += out.grad * (other * self.data ** (other - 1))

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "relu")

        # f(out(self)) = relu(self)
        # -> f'(out) * out'(self)
        # out'(self) -> the relu doesn't change the slope of the function, i.e. its gradient is 1.0 if the value is > 0; and 0 otherwise
        def _backward():
            self.grad += out.grad * (1.0 if out.data > 0 else 0)

        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.data

        def sig(x):
            return 1 / (1 + math.exp(-x))

        out = Value(sig(x), (self,), "\u03C3")

        # f(out(self)) = 1 / (1 + math.exp(-self.data))
        # -> f'(out) * out'(self)
        # out'(self) -> sig(x) * (1 - sig(x)). see proof on Goodnotes
        def _backward():
            self.grad += out.grad * sig(x) * (1 - sig(x))

        out._backward = _backward

        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
