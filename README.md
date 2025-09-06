# NN_foundation

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
