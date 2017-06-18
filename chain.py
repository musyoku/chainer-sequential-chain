import chainer

class Chain(chainer.Chain):
	def __init__(self, *layers):
		super(Chain, self).__init__()
		assert len(layers) > 0
		assert not hasattr(self, "layers")
		self.layers = layers
		with self.init_scope():
			for idx, layer in enumerate(layers):
				if isinstance(layer, chainer.Link):
					setattr(self, "layer_%d" % idx, layer)

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		return x