# PyTorch Official Tutorial Series — Study Notes

### For Recommender Systems & Interview Prep

---

## Table of Contents

1. [Notebook 1 — PyTorch Tensors (Basic)](#nb1)
2. [Notebook 2 — PyTorch Tensors (Deep Dive)](#nb2)
3. [Notebook 3 — Autograd](#nb3)
4. [Notebook 4 — Building Models in PyTorch](#nb4)
5. [Notebook 5 — A Simple PyTorch Model (LeNet)](#nb5)
6. [Notebook 6 — Dataset and DataLoader](#nb6)
7. [Notebook 7 — A Simple PyTorch Training Loop](#nb7)
8. [Notebook 8 — TensorBoard Support in PyTorch](#nb8)
9. [Notebook 9 — Getting Started with Captum](#nb9)

---

## <a id="nb1"></a>Notebook 1 — PyTorch Tensors (Basic)

**File:** `1_-_PyTorch_Tensors.ipynb`

### What is a Tensor?

A **tensor** is the fundamental data structure in PyTorch — think of it as a multi-dimensional array (like NumPy's ndarray, but with GPU support and autograd capabilities).

| Dimensions | Common Name |
| ---------- | ----------- |
| 0-D        | Scalar      |
| 1-D        | Vector      |
| 2-D        | Matrix      |
| 3-D+       | Tensor      |

### Creating Tensors — Key Methods

```python
import torch

# All-zeros (default dtype = float32)
z = torch.zeros(5, 3)

# All-ones with explicit dtype
i = torch.ones((5, 3), dtype=torch.int16)

# Random tensors (reproducible with seed)
torch.manual_seed(1729)
r = torch.rand(2, 2)  # uniform [0, 1)
```

> **Interview Point:** `torch.zeros()` returns `float32` by default. When you print a non-default dtype, PyTorch shows it explicitly. This is a subtle but useful debugging hint.

### Random Seeds — Why They Matter

```python
torch.manual_seed(1729)
r1 = torch.rand(2, 2)  # same values every time
r2 = torch.rand(2, 2)  # different (seed consumed)

torch.manual_seed(1729)
r3 = torch.rand(2, 2)  # same as r1 again!
```

**ELI5:** A random seed is like a recipe. If you start with the same recipe number, you always bake the same cake.

> **Interview Point:** Reproducibility is critical in ML research. Always set seeds when you need deterministic experiments. How does setting random seed generate same values? - Key idea is A seed fixes the starting point of the Pseudo Random Number Generator (PRNG). The PRNG then deterministically generates a sequence of values while updating its internal state.
> Same seed + Same sequence of random operations -> same values

### Arithmetic Operations

Operations on tensors with the **same shape** work element-wise:

```python
ones = torch.ones(2, 3)
twos = torch.ones(2, 3) * 2
threes = ones + twos    # element-wise addition
print(threes.shape)     # torch.Size([2, 3])
```

**Shape mismatch = RuntimeError** — PyTorch will throw an error if shapes don't match.

### Mathematical Operations (Sample)

```python
r = torch.rand(2, 2) - 0.5 * 2  # values in [-1, 1]

torch.abs(r)          # absolute value
torch.asin(r)         # inverse sine
torch.det(r)          # determinant
torch.svd(r)          # singular value decomposition
torch.std_mean(r)     # (std, mean) tuple
torch.max(r)          # maximum scalar
```

> **Interview Point:** PyTorch has 300+ tensor operations. For recommender systems, key ones include matrix multiplication (`torch.matmul`), dot products, and norms.

---

## <a id="nb2"></a>Notebook 2 — PyTorch Tensors (Deep Dive)

**File:** `Video_2_-_Tensors.ipynb`

### All Ways to Create Tensors

```python
# Uninitialized (whatever is in memory)
x = torch.empty(3, 4)

# From Python data
some_constants = torch.tensor([[3.1416, 2.718], [1.618, 0.007]])
some_integers = torch.tensor((2, 3, 5, 7, 11))  # tuples work too

# Match another tensor's shape
zeros_like_x = torch.zeros_like(x)
ones_like_x  = torch.ones_like(x)
rand_like_x  = torch.rand_like(x)
```

> **Note:** `torch.tensor()` creates a **copy** of the data. This is different from `torch.from_numpy()` which shares memory.

### Data Types

```python
a = torch.ones((2, 3), dtype=torch.int16)

# Convert with .to()
b = torch.rand((2, 3), dtype=torch.float64) * 20.
c = b.to(torch.int32)  # truncates (no rounding!)
```

**Available dtypes:** `torch.bool`, `torch.int8`, `torch.uint8`, `torch.int16`, `torch.int32`, `torch.int64`, `torch.half` (float16), `torch.float` (float32), `torch.double` (float64), `torch.bfloat16`

> **Interview Point:** For production ML, `float32` is standard. `float16`/`bfloat16` are used for GPU memory efficiency (mixed-precision training).

### Broadcasting — The Golden Rule

Broadcasting lets you operate on tensors of _different but compatible shapes_.

**Rules (check from the last dimension, going left):**

- Each dimension must be **equal**, OR
- One of the dimensions must be **1**, OR
- The dimension **doesn't exist** in one tensor

```python
# WORKS: (2, 4) * (1, 4) — last dims match, first dim 1 broadcasts
rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)  # each row multiplied

# WORKS: (4, 3, 2) * (3, 2) — missing first dim broadcasts
a = torch.ones(4, 3, 2)
b = a * torch.rand(3, 2)

# FAILS: (4, 3, 2) * (4, 3) — dimensions checked right-to-left: 2 ≠ 3
# RuntimeError!
```

**ELI5:** Broadcasting is like telling the smaller tensor to "repeat itself" to fill in the gaps, as long as the shapes are compatible.

> **Interview Point:** Broadcasting is essential for recommender systems — e.g., multiplying a user embedding matrix `(num_users, latent_dim)` by an item embedding `(1, latent_dim)`.

### In-Place Operations (the `_` suffix)

```python
a = torch.tensor([0, π/4, π/2])
torch.sin(a)    # creates NEW tensor, a unchanged
torch.sin_(a)   # modifies a IN-PLACE

# On instances:
a.add_(b)   # a += b, modifies a
b.mul_(b)   # b *= b, modifies b
```

> **⚠️ Warning:** In-place ops can cause issues with autograd (gradient tracking). Avoid on leaf tensors that require gradients.

### Copying Tensors

```python
a = torch.ones(2, 2)
b = a           # NOT a copy! b is just another name for a
a[0][1] = 561   # changes BOTH a and b

# Real copy:
b = a.clone()   # separate memory
```

**`.detach()` vs `.clone()`:**

- `.clone()` — copies data + keeps autograd history
- `.detach()` — creates new tensor with NO gradient history
- `.detach().clone()` — copy with no grad tracking (use for metrics/logging)

### Moving to GPU

```python
# Check GPU availability
if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')

# Create on device
x = torch.rand(2, 2, device=my_device)

# Move existing tensor
y = torch.rand(2, 2)
y = y.to(my_device)
```

> **⚠️ Critical:** All tensors in an operation **must be on the same device**. Mixing CPU and GPU tensors throws a RuntimeError.

### Shape Manipulation

```python
# unsqueeze: add a dimension of size 1
a = torch.rand(3, 226, 226)        # single image
b = a.unsqueeze(0)                  # add batch dim → (1, 3, 226, 226)

# squeeze: remove dimensions of size 1
a = torch.rand(1, 20)               # (1, 20)
b = a.squeeze(0)                    # → (20,)
# Note: squeeze only removes dims of size 1!

# reshape: change shape while keeping total elements same
output3d = torch.rand(6, 20, 20)
input1d = output3d.reshape(6 * 20 * 20)  # → (2400,)
```

> **Interview Point:** `reshape()` returns a **view** when possible (shares memory). Use `.clone()` after reshape if you need independent data.

### NumPy Bridge

```python
import numpy as np

# NumPy → PyTorch (shared memory!)
numpy_array = np.ones((2, 3))
pytorch_tensor = torch.from_numpy(numpy_array)
# Changing numpy_array ALSO changes pytorch_tensor

# PyTorch → NumPy (shared memory!)
pytorch_rand = torch.rand(2, 3)
numpy_rand = pytorch_rand.numpy()
```

> **⚠️ Warning:** The shared memory means changes to one affect the other. Use `.copy()` on the NumPy side or `.clone()` on the PyTorch side if you need independence.

---

## <a id="nb3"></a>Notebook 3 — Autograd

**File:** `Video_3_-_Autograd.ipynb`

### What is Autograd?

Autograd is PyTorch's **automatic differentiation engine**. It dynamically tracks computations at runtime and computes gradients automatically — essential for backpropagation in neural network training.

**Why we need it:** Training a neural network requires computing how much each parameter contributes to the loss (via partial derivatives). Autograd does this automatically, even for complex, branching computation graphs.

### The Math (Simplified)

For a model M with inputs **x** and loss L:

- We want `∂L/∂weights` (how much each weight affects the loss)
- By chain rule: `∂L/∂x = (∂L/∂y)(∂y/∂x)` where y = M(x)
- Autograd computes this product across the entire computation graph

**ELI5:** Imagine you're adjusting a recipe. The loss is how bad the cake tastes. Autograd tells you exactly which ingredient (parameter) to change and by how much.

### Enabling Gradient Tracking

```python
a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
# Setting requires_grad=True tells PyTorch: "track everything done to this tensor"

b = torch.sin(a)
print(b)  # shows: grad_fn=<SinBackward>
# The grad_fn records what operation created this tensor
```

### Computation Graph

Every tensor with `requires_grad=True` tracks a history:

```python
b = torch.sin(a)      # grad_fn=<SinBackward>
c = 2 * b             # grad_fn=<MulBackward0>
d = c + 1             # grad_fn=<AddBackward0>
out = d.sum()         # grad_fn=<SumBackward0>

# Walk the graph:
print(d.grad_fn)                              # AddBackward0
print(d.grad_fn.next_functions)               # points to MulBackward0
# ... chain continues all the way back to 'a' (which has no grad_fn)
print(a.grad_fn)                              # None (leaf node)
```

### Computing Gradients — `.backward()`

```python
out.backward()     # trigger backpropagation
print(a.grad)      # gradient of 'out' with respect to 'a'
# Result: 2 * cos(a) — chain rule: d(sum(2*sin(a)+1))/da = 2*cos(a)
```

> **Key Rule:** Only **leaf nodes** (the original inputs) accumulate gradients. Intermediate tensors do not.

### Autograd in Model Training

```python
model = TinyModel()   # nn.Module subclass (weights auto-tracked)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Forward pass
prediction = model(some_input)
loss = (ideal_output - prediction).pow(2).sum()

# Before backward pass: weights unchanged, no gradients yet
print(model.layer2.weight.grad)  # None

# Backward pass
loss.backward()

# Gradients computed, but weights still the SAME
print(model.layer2.weight.grad[0][0:10])  # non-zero gradients now

# Update weights
optimizer.step()
print(model.layer2.weight[0][0:10])  # weights have changed!

# ⚠️ CRUCIAL: Zero gradients before next batch!
optimizer.zero_grad()
# Without this, gradients ACCUMULATE across batches → wrong learning!
```

> **Interview Point: `optimizer.zero_grad()` is one of the most common mistakes for beginners.** Forgetting it causes gradient accumulation, making the loss explode or not converge.

### Turning Autograd Off

Three ways:

```python
# Method 1: Change requires_grad flag
a.requires_grad = False
b2 = 2 * a  # no grad tracking

# Method 2: Context manager (recommended for validation)
with torch.no_grad():
    c2 = a + b  # no tracking within this block

# Method 3: Decorator
@torch.no_grad()
def add_tensors2(x, y):
    return x + y  # no tracking

# Method 4: detach() for metric logging
x = torch.rand(5, requires_grad=True)
y = x.detach()  # same data, no grad history
```

> **Interview Point:** Always wrap your **validation/test loop** with `torch.no_grad()` — it speeds up computation and saves memory by not building the computation graph.

### ⚠️ In-Place Operations + Autograd

```python
a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
torch.sin_(a)  # RuntimeError!
# "a leaf Variable that requires grad is being used in an in-place operation"
```

**Why?** Autograd needs the original values for gradient computation. In-place ops destroy them.

### Advanced: Jacobian & Vector-Jacobian Products

```python
# Jacobian of a function
def exp_adder(x, y):
    return 2 * x.exp() + 3 * y

inputs = (torch.rand(1), torch.rand(1))
torch.autograd.functional.jacobian(exp_adder, inputs)
# Returns matrix of all partial derivatives

# Vector-Jacobian product (more efficient for backprop)
torch.autograd.functional.vjp(do_some_doubling, inputs, v=my_gradients)
```

---

## <a id="nb4"></a>Notebook 4 — Building Models in PyTorch

**File:** `Video_4_-_Building_Models_in_PyTorch.ipynb`

### The `torch.nn.Module` Class

Every PyTorch model inherits from `torch.nn.Module`. It:

- Manages learnable **parameters** (`torch.nn.Parameter`)
- Enables parameter iteration with `.parameters()`
- Provides clean string representation
- Handles device placement, serialization, etc.

```python
class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()
print(tinymodel)         # shows architecture
print(tinymodel.linear2) # inspect individual layer

# Iterate ALL parameters
for param in tinymodel.parameters():
    print(param)

# Iterate ONE layer's parameters
for param in tinymodel.linear2.parameters():
    print(param)
```

> **Interview Point:** `nn.Module` automatically registers `nn.Parameter` objects assigned as attributes. This is why you don't need to manually call `requires_grad=True` on weights — it's set by default.

### Common Layer Types

#### 1. Linear Layers (Fully Connected)

```python
lin = torch.nn.Linear(3, 2)   # 3 inputs, 2 outputs
x = torch.rand(1, 3)
y = lin(x)   # y = xW^T + b
```

- Weights are `Parameters` (auto-tracked by autograd)
- Output: `(batch_size, out_features)`

#### 2. Convolutional Layers

```python
self.conv1 = torch.nn.Conv2d(1, 6, 5)
# Args: (in_channels, out_channels, kernel_size)
# For color images: in_channels=3
```

- Scans for spatial patterns
- Output: activation map of shape `(batch, channels, H, W)`
- After 5×5 kernel on 32×32 image → 28×28 activation map

#### 3. Recurrent Layers (LSTM)

```python
self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
# Maintains hidden state across time steps
# Used for sequential data: text, time series
```

#### 4. Transformer

```python
# PyTorch has torch.nn.Transformer built-in
# Configure attention heads, encoder/decoder layers, etc.
```

### Data Manipulation Layers (No Learning)

#### Max Pooling

```python
maxpool_layer = torch.nn.MaxPool2d(3)  # 3×3 window
# Reduces spatial size by taking max in each window
# 6×6 input → 2×2 output (with 3×3 stride-3 pooling)
```

#### Batch Normalization

```python
norm_layer = torch.nn.BatchNorm1d(4)
normed_tensor = norm_layer(my_tensor)
# Centers data around 0, std ≈ 1
# Enables higher learning rates, prevents vanishing gradients
```

**ELI5:** BatchNorm is like re-centering the data after every layer. It prevents the internal values from drifting too high or low, making training more stable.

#### Dropout

```python
dropout = torch.nn.Dropout(p=0.4)  # 40% chance of zeroing each value
output = dropout(my_tensor)
# Only active during TRAINING — disabled automatically during eval
```

**Purpose:** Forces the model to learn robust features that don't depend on any single neuron. Prevents overfitting.

> **Interview Point:** Always call `model.train()` before training and `model.eval()` before validation. This switches Dropout and BatchNorm to their correct modes.

### Activation Functions

Activation functions introduce **non-linearity** — without them, stacking layers is mathematically equivalent to a single linear transformation.

| Activation | When to Use                        |
| ---------- | ---------------------------------- |
| `ReLU`     | Standard hidden layers             |
| `Sigmoid`  | Binary classification output       |
| `Softmax`  | Multi-class output (probabilities) |
| `Tanh`     | When values need to be in [-1, 1]  |

### Loss Functions

```python
nn.CrossEntropyLoss()    # classification (combines softmax + NLL)
nn.MSELoss()             # regression (mean squared error)
nn.NLLLoss()             # negative log-likelihood
```

> **Interview Point for Recommender Systems:** For collaborative filtering, `nn.MSELoss()` is common for rating prediction. For implicit feedback ranking, BPR loss or cross-entropy on sampled negatives is typical.

---

## <a id="nb5"></a>Notebook 5 — A Simple PyTorch Model (LeNet)

**File:** `2_-_A_Simple_PyTorch_model.ipynb`

### LeNet-5 Architecture

LeNet-5 is one of the first successful CNNs, designed for handwritten digit recognition.

```
Input (1×32×32)
    ↓
C1: Conv2d(1, 6, 3)   + ReLU + MaxPool
    ↓ (6×14×14)
C3: Conv2d(6, 16, 3)  + ReLU + MaxPool
    ↓ (16×6×6)
Flatten → (576,)
    ↓
F5: Linear(576, 120)  + ReLU
    ↓
F6: Linear(120, 84)   + ReLU
    ↓
OUT: Linear(84, 10)
```

```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image math
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # exclude batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

### Key Structural Concepts

1. **Batch dimension:** PyTorch always expects `(batch_size, ...)`. Even a single image needs shape `(1, 1, 32, 32)`.
2. **`view(-1, n)`:** The `-1` means "infer this dimension from total elements". Used to flatten conv output for linear layers.
3. **`net(input)` not `net.forward(input)`:** Always call the model like a function — `net(input)`. This ensures hooks and other PyTorch internals are triggered.
4. **Output shape matches batch:** For 16 images, output is `(16, 10)`.

> **Interview Point:** The architecture pattern Conv→Pool→Conv→Pool→Flatten→FC→FC→Output is the backbone of most CNNs. Understanding this prepares you to interpret and build larger models like ResNet, VGG, etc.

---

## <a id="nb6"></a>Notebook 6 — Dataset and DataLoader

**File:** `3_-_Dataset_and_DataLoader.ipynb`

### Data Pipeline Overview

```
Raw Data → Dataset → DataLoader → Model
```

### Transforms

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),          # PIL Image → Tensor, scales [0,255] → [0,1]
    transforms.Normalize(
        (0.5, 0.5, 0.5),           # mean per channel
        (0.5, 0.5, 0.5)            # std per channel
    )
    # Result: values in [-1, 1]
])
```

**Why Normalize?** Most activation functions (ReLU, Tanh) have the strongest gradients near 0. Centering data speeds up learning.

### Dataset

```python
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform   # applied to each sample
)
```

**Built-in Datasets:** CIFAR10/100, MNIST, FashionMNIST, ImageNet, CelebA, etc.

**Custom Dataset:** Subclass `torch.utils.data.Dataset` and implement `__len__` and `__getitem__`.

### DataLoader

```python
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,    # samples per batch
    shuffle=True,    # randomize order each epoch
    num_workers=2    # parallel data loading workers
)
```

| Parameter     | Purpose                                             |
| ------------- | --------------------------------------------------- |
| `batch_size`  | Samples per forward pass                            |
| `shuffle`     | Randomize order (True for training, False for test) |
| `num_workers` | Parallel CPU workers for loading                    |
| `drop_last`   | Drop incomplete last batch                          |

### Using the DataLoader

```python
dataiter = iter(trainloader)
images, labels = dataiter.next()  # or next(dataiter)

# Shape: images = (4, 3, 32, 32), labels = (4,)
```

### Key Conceptual Distinction

- **`Dataset`** knows about the **data** (how to load/transform individual samples)
- **`DataLoader`** knows about **batching** (how to assemble batches for training)

> **Interview Point:** For recommender systems, you'll write custom `Dataset` classes. A user-item interaction dataset would return `(user_id, item_id, rating)` tuples. DataLoader handles batching automatically.

---

## <a id="nb7"></a>Notebook 7 — A Simple PyTorch Training Loop

**File:** `4_-_A_Simple_PyTorch_Training_Loop.ipynb`

### The Complete Training Loop Pattern

This is the most important pattern in all of PyTorch:

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # ① Zero gradients (CRITICAL — prevents accumulation)
        optimizer.zero_grad()

        # ② Forward pass
        outputs = net(inputs)

        # ③ Compute loss
        loss = criterion(outputs, labels)

        # ④ Backward pass (compute gradients)
        loss.backward()

        # ⑤ Update weights
        optimizer.step()

        # Logging
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss/2000:.3f}')
            running_loss = 0.0

print('Finished Training')
```

### The 5 Steps — Memorize This!

| Step               | What Happens                                  |
| ------------------ | --------------------------------------------- |
| `zero_grad()`      | Clear gradient buffers from previous batch    |
| `net(inputs)`      | Forward pass — compute predictions            |
| `criterion(...)`   | Compute loss (how wrong we are)               |
| `loss.backward()`  | Backprop — compute gradients                  |
| `optimizer.step()` | Update weights in direction that reduces loss |

**ELI5:** It's like steering a car:

1. Check you're not accidentally holding the wheel from last turn (zero_grad)
2. Look ahead (forward pass)
3. Check how far off course you are (loss)
4. Decide which way to turn (backward)
5. Turn the wheel (optimizer.step)

### Loss Curve Interpretation

The training output showed monotonically decreasing loss:

```
[1,  2000] loss: 2.234
[1,  4000] loss: 1.876
...
[2, 12000] loss: 1.273
```

✅ This is **good** — the model is learning.

### Testing for Generalization

```python
correct = 0
total = 0

with torch.no_grad():  # no gradients needed for evaluation
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # argmax
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
```

- Result: ~55% accuracy (vs. 10% random baseline for 10 classes)
- **Overfitting:** If test accuracy << training accuracy, model memorized training data

> **Interview Point:** The difference between `loss.item()` and just `loss` is important. `.item()` extracts a plain Python number and **detaches the computation graph**, preventing memory leaks in logging loops.

---

## <a id="nb8"></a>Notebook 8 — TensorBoard Support in PyTorch

**File:** `Video_5_-_Tensorboard_Support_in_PyTorch.ipynb`

### What is TensorBoard?

TensorBoard is a **visualization toolkit** (originally from TensorFlow, supported by PyTorch) that lets you:

- Monitor training/validation loss curves in real-time
- Visualize model architecture as a graph
- Display images, embeddings, and custom scalars

### Setup

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/fashion_mnist_experiment_1')
```

Start TensorBoard:

```bash
tensorboard --logdir=runs
# Open http://localhost:6006/ in browser
```

### Logging Images

```python
img_grid = torchvision.utils.make_grid(images)
writer.add_image('Four Fashion-MNIST Images', img_grid)
writer.flush()  # write to disk immediately
```

### Logging Scalars (Loss Curves)

```python
writer.add_scalars(
    'Training vs. Validation Loss',
    {'Training': avg_loss, 'Validation': avg_vloss},
    epoch * len(training_loader) + i   # global step
)
```

### Visualizing Model Architecture

```python
writer.add_graph(net, images)  # traces the model with sample input
writer.flush()
```

### Visualizing Embeddings (Projector)

```python
features = images.view(-1, 28 * 28)  # flatten images
writer.add_embedding(
    features,
    metadata=class_labels,    # color-code by class
    label_img=images.unsqueeze(1)
)
writer.flush()
writer.close()  # always close when done
```

This creates a 3D PCA/t-SNE visualization of your data, showing how the model groups similar items.

### The Net Used (Fashion-MNIST LeNet Variant)

```
Conv2d(1, 6, 5) → MaxPool → Conv2d(6, 16, 5) → MaxPool
→ Linear(256, 120) → Linear(120, 84) → Linear(84, 10)
```

Trained with:

- **Loss:** CrossEntropyLoss
- **Optimizer:** SGD (lr=0.001, momentum=0.9)
- Validation loss logged every 1000 mini-batches

> **Interview Point for Recommender Systems:** TensorBoard's embedding projector is extremely useful for inspecting learned user/item embeddings. If similar items cluster together, your model has learned meaningful representations.

---

## <a id="nb9"></a>Notebook 9 — Getting Started with Captum (Model Interpretability)

**File:** `Getting-Started-with-Captum.ipynb`

### What is Captum?

Captum (Latin: "comprehension") is PyTorch's model interpretability library. It helps answer: **"Why did the model make this prediction?"**

### Three Types of Attribution

| Type                    | Question Answered                       | Example                                    |
| ----------------------- | --------------------------------------- | ------------------------------------------ |
| **Feature Attribution** | Which input features drove this output? | Which pixels made the model predict "cat"? |
| **Layer Attribution**   | How did a hidden layer respond?         | What did conv3 activate on?                |
| **Neuron Attribution**  | How did a single neuron respond?        | What input maximally activates neuron #42? |

### Two Categories of Algorithms

#### Gradient-Based (analytical)

Compute derivatives of output w.r.t. input:

- `IntegratedGradients` — approximates integral of gradients along a path from baseline to input
- `LayerGradCam` — gradients × activations in a conv layer
- `NeuronConductance`

#### Perturbation-Based (experimental)

Change the input and measure the effect:

- `Occlusion` — mask regions and see how output changes
- `FeatureAblation`
- `FeaturePermutation`

### Feature Attribution with Integrated Gradients

```python
from captum.attr import IntegratedGradients

# Load pretrained ResNet-101
model = models.resnet101(pretrained=True).eval()

# Get prediction
output = model(input_img)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)

# Compute attributions
integrated_gradients = IntegratedGradients(model)
attributions_ig = integrated_gradients.attribute(
    input_img,
    target=pred_label_idx,
    n_steps=200   # number of integration steps (more = more accurate)
)

# Visualize (heat map over original image)
viz.visualize_image_attr(
    np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
    np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
    method='heat_map',
    sign='positive'
)
```

**ELI5:** Integrated Gradients asks "what path from blank (baseline) to the actual image contributes most to the model's confidence?" The regions with high gradient integral are most responsible for the prediction.

### Feature Attribution with Occlusion

```python
from captum.attr import Occlusion

occlusion = Occlusion(model)
attributions_occ = occlusion.attribute(
    input_img,
    target=pred_label_idx,
    strides=(3, 8, 8),                # step size
    sliding_window_shapes=(3, 15, 15), # mask size
    baselines=0                        # replace with 0s
)
```

**ELI5:** Cover different parts of the image with a black square and see which regions, when covered, most reduce the model's confidence.

### Layer Attribution with GradCAM

```python
from captum.attr import LayerGradCam, LayerAttribution

layer_gradcam = LayerGradCam(model, model.layer3[1].conv2)
attributions_lgc = layer_gradcam.attribute(input_img, target=pred_label_idx)

# Upsample to match input image size
upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input_img.shape[2:])
```

**GradCAM Process:**

1. Compute gradients of output w.r.t. a specific conv layer
2. Average gradients over each output channel → importance weights
3. Weighted sum of activations → importance map
4. Shows **where** in the image the model is "looking"

### Captum Insights (Interactive Widget)

```python
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

visualizer = AttributionVisualizer(
    models=[model],
    score_func=lambda o: torch.nn.functional.softmax(o, 1),
    classes=list(map(lambda k: idx_to_labels[k][1], idx_to_labels.keys())),
    features=[ImageFeature("Photo", baseline_transforms=[baseline_func], input_transforms=[])],
    dataset=[Batch(input_imgs, labels=[282, 849, 69])]
)

visualizer.render()  # launches interactive widget in Jupyter
```

> **Interview Point for Recommender Systems:** Captum-style interpretability for recommender systems means understanding why item X was recommended to user Y. You can apply Integrated Gradients to highlight which user features or interaction history most influenced the recommendation.

---

## Key Themes & Interview Topics Summary

### The Core PyTorch Training Loop (Most Important!)

```
for epoch in epochs:
  for batch in dataloader:
    optimizer.zero_grad()  ← don't forget!
    output = model(input)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
```

### Critical Gotchas

| Mistake                                     | Consequence                           |
| ------------------------------------------- | ------------------------------------- |
| Forgetting `zero_grad()`                    | Gradients accumulate → wrong learning |
| Not using `torch.no_grad()` in eval         | Wastes memory, slower                 |
| Forgetting `model.eval()` before eval       | Dropout/BN behave as training         |
| Mixing CPU/GPU tensors                      | RuntimeError                          |
| In-place ops on autograd leaves             | RuntimeError                          |
| Not calling `writer.close()` in TensorBoard | Missing data                          |

### Tensor Operations — Cheatsheet

```python
# Creation
torch.zeros(m, n)          torch.ones(m, n)
torch.rand(m, n)           torch.tensor([1, 2, 3])
torch.zeros_like(t)        torch.from_numpy(arr)

# Shape
t.shape                    t.view(-1, n)
t.unsqueeze(0)             t.squeeze(0)
t.reshape(new_shape)

# Operations
t.to(device)               t.to(dtype)
t.clone()                  t.detach()
t.detach().clone()

# Math
torch.matmul(a, b)         a @ b  # same as matmul
torch.max(t)               torch.mean(t)
torch.sum(t)               torch.std(t)
```

### For Recommender Systems Specifically

1. **Embeddings:** `nn.Embedding(num_items, embedding_dim)` — learn item/user representations
2. **Matrix Factorization:** Dot product of user and item embeddings
3. **Batching:** Custom `Dataset` with `(user_id, item_id, rating)` tuples
4. **Loss:** MSE for explicit ratings; BPR or BCE for implicit feedback
5. **Evaluation:** Wrap test loop in `torch.no_grad()`
6. **Interpretability:** Captum to understand why items were recommended

---

_Notes compiled from the official PyTorch tutorial series. For deeper dives, see [pytorch.org/tutorials](https://pytorch.org/tutorials)._
