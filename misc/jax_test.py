import jax.numpy as np
from jax import grad, jit, vmap

def predict(params, inputs):
  for W, b in params:
    print(W,b, inputs)
    outputs = np.dot(inputs, W) + b
    inputs = np.tanh(outputs)
  return outputs

def logprob_fun(params, inputs, targets):
  preds = predict(params, inputs)
  return np.sum((preds - targets)**2)

grad_fun = jit(grad(logprob_fun))  # compiled gradient evaluation function
perex_grads = jit(vmap(grad_fun, in_axes=(None, 0, 0)))  # fast per-example grads


if __name__ == '__main__':
    params = [(np.asarray([1.0,1.0,1.0]),np.asarray([1.0]))]
    targets = np.asarray([1.0,1.0,1.0])
    grad_fun = jit(grad(logprob_fun))
    perex_grads = jit(vmap(grad_fun, in_axes=(None, 0, 0)))
    grad_fun(params, targets, np.asarray([1,1,1]))