class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op

	def __repr__(self):
		def fmt(x):
			return int(x) if float(x).is_integer() else round(x, 4)
		return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

	def _cast_Value(self, other, *args, **kwargs):
		if not isinstance(other, Value):
			return Value(other)
		return other

	def __add__(self, other):
		other = self._cast_Value(other)
		out = Value(self.data + other.data, (self, other), '+')

		def _step():
			self.grad += out.grad
			other.grad += out.grad
		
		out._backward = _step
		return out

	def __radd__(self, other):
		return self.__add__(other)

	def __mul__(self, other):
		other = self._cast_Value(other)

		out = Value(self.data * other.data, (self, other), '*')

		def _step():
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad
		
		out._backward = _step
		return out

	def __rmul__(self, other):
		return self.__mul__(other)

	def relu(self):
		out = Value(max(0, self.data), (self, ), 'relu')

		def _step():
			self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

		out._backward = _step
		return out

	def backward(self):
		topo, visited = list(), set()
		
		def _build_topo(node):
			if node in visited:
				return None
			visited.add(node)
			for neigh in node._prev:
				_build_topo(neigh)
			topo.append(node)
		_build_topo(self)

		self.grad = 1.0
		for node in reversed(topo):
			node._backward()