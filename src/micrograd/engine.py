
class Value:
    def __init__(self, data, _children = (), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        return self + (-1*other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int and float for now"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += out.grad*other*self.data**(other - 1)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self*other**-1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), _op='exp', label='exp(x)')

        def _backward():
            self.grad += out.grad*out.data
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        out = Value(np.tanh(x), (self,), _op='tanh', label='tanh(x)')

        def _backward():
            self.grad += (1 - out.data**2)*out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
          if v not in visited:
            visited.add(v)
            for child in v._prev:
              build_topo(child)
            topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
          node._backward()
