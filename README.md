# Sequential Chain for Chainer

### New way

```
from chain import Chain

model = Chain(
	L.Linear(None, 1024),
	F.leaky_relu,
	L.BatchNormalization(1024),
	L.Linear(None, 512),
	F.relu,
	L.BatchNormalization(512),
	L.Linear(None, 256),
	F.elu,
	L.BatchNormalization(256),
	L.Linear(None, 128),
	F.tanh,
	L.BatchNormalization(128),
	L.Linear(None, 10),
)
y = model(x)
```

### Official way

```
class Chain(chainer.Chain):
	def __init__(self):
		super(Chain, self).__init__()
		with self.init_scope():
			self.l1 = L.Linear(None, 1024)
			self.l2 = L.Linear(None, 512)
			self.l3 = L.Linear(None, 256)
			self.l4 = L.Linear(None, 128)
			self.l5 = L.Linear(None, 10)
			self.bn1 = L.BatchNormalization(1024)
			self.bn2 = L.BatchNormalization(512)
			self.bn3 = L.BatchNormalization(256)
			self.bn4 = L.BatchNormalization(128)

	def __call__(self, x):
		out = self.l1(x)
		out = F.leaky_relu(out)
		out = self.bn1(out)
		out = self.l2(out)
		out = F.relu(out)
		out = self.bn2(out)
		out = self.l3(out)
		out = F.elu(out)
		out = self.bn3(out)
		out = self.l4(out)
		out = F.relu(out)
		out = self.bn4(out)
		out = self.l5(out)
		return out

model = Chain()
y = model(x)
```