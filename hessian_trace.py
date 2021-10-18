import jax
import jax.numpy as jnp


def rademacher_tree(rng, x):

  def random_leaf(leaf):
    #rng, subkey = jax.random.split(rng, 2)
    return jax.random.rademacher(rng, shape=leaf.shape, dtype=leaf.dtype)

  return jax.tree_util.tree_map(random_leaf, x)


def hessian_trace(key, params, grads, maxIter=50, tol=1e-3):
  """
  Trace 
  compute the trace of hessian using Hutchinson's method
  maxIter: maximum iterations used to compute trace
  tol: the relative tolerance
  """
  v = jax.random.rademacher(rng, shape=params.shape, dtype=params.dtype)

  Hv = hessian_vector_product(grads, params, v)
  tangents = rademacher_tree(key, x)


  (_, hessian_vector_prod) = jax.jvp(jax.grad(f, has_aux=True), x, tangents)

  hessian = jax.tree_util.multi_treemap(lambda x, y: x*y, tangents, hessian_vector_prod)

  return jnp.mean(hessian)

  # trace_vhv = []
  # trace = 0.

  # for i in range(maxIter):
  #   key, subkey = jax.random.split(key, 2)

  #   # generate Rademacher random variables (+1, -1)
  #   jax.tree_util.tree_map( , params)

  #   v = [
  #       torch.randint_like(p, high=2, device=device)
  #       for p in params
  #   ]
    

    
  #   Hv = hessian_vector_product(grads, params, v)

  #   Hv = jax.tree_util.multi_treemap(lambda x, y: jnp.dot(x,y),  )
  #   trace_vhv.append(group_product(Hv, v).cpu().item())
  #   if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
  #       return trace_vhv
  #   else:
  #       trace = np.mean(trace_vhv)

  # return trace_vhv