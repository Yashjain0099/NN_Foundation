# NN_foundation for EveryOne

## Day 1 - Vectors, Matrices, and Tensors

**Concepts:**
- Vector = 1D array
- Matrix = 2D array
- Tensor = nD array

**Hands-on:**
- Dot product (manual vs NumPy)
- Matrix multiplication (manual vs NumPy)
- Broadcasting example

👉 [Notebook Link](notebooks/day1_vectors_matrices_tensors.ipynb)


## Day 2 - Derivatives, Gradients, and Gradient Checking

**Concepts:**
- Derivative = rate of change (slope)
- Chain rule = used in backpropagation
- Gradient = vector of partial derivatives
- Gradient checking = sanity check

**Hands-on:**
- Computed derivative of f(w) = w^2 + 3w + 1 at w=2
- Verified using finite difference method
- Wrote general Python function for numerical derivative

👉 [Notebook Link](notebooks/calculus_for_learning.ipynb)

## Day 3 - Logistic Regression

**Concepts:**
- Logistic regression predicts probability using sigmoid.
- Loss = binary cross-entropy.
- Parameters updated via gradient descent.

**Hands-on:**
- Implemented logistic regression from scratch (NumPy).
- Plotted decision boundary + loss curve.
- Compared with sklearn’s LogisticRegression.

👉 [Notebook Link](notebooks/LossFunction_LogisticRegression.ipynb)


## Day 4 - Activation Functions

**Concepts:**
- Sigmoid, Tanh, ReLU, Softmax
- Gradients explain vanishing gradient problem
- ReLU helps mitigate vanishing gradient for deep nets

**Hands-on:**
- Implemented activations + gradients in NumPy
- Plotted functions over [-10, 10]
- Compared gradients → saw vanishing in sigmoid/tanh

👉 [Notebook Link](notebooks/day4_activation_functions.ipynb)


## Day 5 - Two-Layer Neural Network (XOR)

**Concepts:**
- Input → Hidden (ReLU) → Output (Sigmoid)
- Forward, loss, backward, update
- Neural nets can solve non-linear problems like XOR

**Hands-on:**
- Built NN from scratch with NumPy
- Implemented forward_propagation, compute_loss, backward_propagation, update_parameters
- Successfully classified XOR dataset

👉 [Notebook Link](notebooks/The_Perceptron_&_1-Hidden-Layer_MLP.ipynb)

## Day 6 - Backpropagation + Gradient Checking

**Concepts:**
- Backpropagation: computing gradients layer by layer
- Gradient checking: comparing analytical vs numerical derivatives
- Loss decreases as parameters update correctly

**Hands-on:**
- Implemented forward, loss, backward, update in NumPy
- Trained NN on `make_moons` dataset
- Verified gradients with numerical approximation
- Saw loss curve dropping as expected 🎯

👉 [Notebook Link](notebooks/Backprapogation.ipynb)
 
