from typing import Callable
import jax
import numpy as np
from functools import partial
import jax.numpy as jnp
from jax._src.lax import lax
from jax._src.tree_util import tree_flatten, tree_unflatten, tree_map
from jax._src import util
from jax._src.lax import slicing
from functional.autograd import fgi_dgaussian

initial_th = 1.0


from jax._src.tree_util import tree_map, tree_structure, \
  tree_unflatten



def S(x, initial_th=initial_th):
  th = initial_th
  phi_init = tree_map(lambda i: jnp.zeros_like(i), x)
  z = tree_map(lambda i, phi: fgi_dgaussian(i - th - phi), x, phi_init)
  phi_new = tree_map(lambda phi_i, zi: phi_i * 0.9 + zi, phi_init, z)
  return phi_new




from jax._src.lax import lax
def _interleave(a, b, axis):
  """Given two Tensors of static shape, interleave them along the first axis."""
  assert a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
  a_pad = [(0, 0, 0)] * a.ndim
  b_pad = [(0, 0, 0)] * b.ndim
  a_pad[axis] = (0, 1 if a.shape[axis] == b.shape[axis] else 0, 1)
  b_pad[axis] = (1, 0 if a.shape[axis] == b.shape[axis] else 1, 1)
  op = lax.bitwise_or if a.dtype == np.bool_ else lax.add
  return op(lax.pad(a, lax._const(a, 0), a_pad),
            lax.pad(b, lax._const(b, 0), b_pad))


def associative_scan(fn: Callable, elems, apply_th: bool = False,
                     aggr_op="max", reverse: bool = False, axis: int = 0):
  if not callable(fn):
    raise TypeError("lax.associative_scan: fn argument should be callable.")
  elems_flat, tree = tree_flatten(elems)

  if reverse:
    elems_flat = [lax.rev(elem, [axis]) for elem in elems_flat]

  # PHI positioning
  def phi_position_finder(x):
      layers = []
      current = x

      # Forward max-pairing
      while True:
          if current.size <= 1:
              break
          even = current[::2]
          odd = current[1::2]
          paired_max = jnp.maximum(even[:odd.size], odd) #

          layers.append(paired_max)
          current = paired_max

      layers.append(jnp.array([], dtype=current.dtype))  # one empty array

      # Backward mean layers
      backward_means = []
      for i in range(len(layers) - 2 - 1, -1, -1):  # skip last two
          curr = layers[i]
          if curr.size > 1:
              means = (curr[:-1] + curr[1:]) // 2
              backward_means.append(means)

      return layers + backward_means

  layer = []

  def combine(a_flat, b_flat):
    a = tree_unflatten(tree, a_flat)
    b = tree_unflatten(tree, b_flat)
    c = fn(a, b)
    c_flat, _ = tree_flatten(c)
    if apply_th:
      phi = S(c)
      phi_flat, _ = tree_flatten(phi)
      layer.append(phi_flat)
      return c_flat, phi
    else:
      return c_flat, None

  axis = util.canonicalize_axis(axis, elems_flat[0].ndim)

  # if not core.is_constant_dim(elems_flat[0].shape[axis]):
  # raise NotImplementedError("associative scan over axis "
  # f"of non-constant size: {elems_flat[0].shape[axis]}. You may be "
  # "able to avoid this on TPU. See b/274176030.")
  num_elems = int(elems_flat[0].shape[axis])
  if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
    raise ValueError('Array inputs to associative_scan must have the same '
                     'first dimension. (saw: {})'
                     .format([elem.shape for elem in elems_flat]))

  def _scan(elems):
    """Perform scan on `elems`."""

    num_elems = elems[0].shape[axis]

    if num_elems < 2:
      return elems

    # Combine adjacent pairs of elements.
    reduced_elems, _ = combine(
      [slicing.slice_in_dim(elem, 0, -1, stride=2, axis=axis) for elem in
       elems],
      [slicing.slice_in_dim(elem, 1, None, stride=2, axis=axis)
       for elem in elems])

    # Recursively compute scan for partially reduced tensors.

    odd_elems = _scan(reduced_elems)

    if num_elems % 2 == 0:
      even_elems, _ = combine(
        [slicing.slice_in_dim(e, 0, -1, axis=axis) for e in odd_elems],
        [slicing.slice_in_dim(e, 2, None, stride=2, axis=axis) for e in elems])
    else:
      even_elems, _ = combine(
        odd_elems,
        [slicing.slice_in_dim(e, 2, None, stride=2, axis=axis) for e in elems])

    even_elems = [
      lax.concatenate([slicing.slice_in_dim(elem, 0, 1, axis=axis), result],
                      dimension=axis)
      for (elem, result) in zip(elems, even_elems)]

    return list(map(partial(_interleave, axis=axis), even_elems, odd_elems))

  scans = _scan(elems_flat)

  if reverse:
    scans = [lax.rev(scanned, [axis]) for scanned in scans]

  if apply_th:
    x = jnp.arange(num_elems)
    output = phi_position_finder(x)
    num_features = len(layer[0])
    feature_padded_layers = [[] for _ in range(num_features)]
    for phi_flat, phi_indices in zip(layer, output):
        for feature_idx, phi_arr in enumerate(phi_flat):
            n_indices = phi_indices.shape[0]
            n_phi = phi_arr.shape[0]
            if n_phi > n_indices:
                phi_arr = phi_arr[:n_indices]
            elif n_phi < n_indices:
                raise ValueError(f"phi_arr rows ({n_phi}) < phi_indices length ({n_indices})")
            D = phi_arr.shape[1]
            phi_padded = jnp.full((num_elems, D), jnp.nan)
            phi_padded = phi_padded.at[phi_indices].set(phi_arr)
            feature_padded_layers[feature_idx].append(phi_padded)
    layer_matrices = [jnp.stack(feature_layers) for feature_layers in feature_padded_layers]


    # Aggregation over phi matrix columns
    if aggr_op == "max":
      col_agg = [jnp.nanmax(mat, axis=0) for mat in layer_matrices]
    elif aggr_op == "mean":
      col_agg = [jnp.nanmean(mat, axis=0) for mat in layer_matrices]
    elif aggr_op == "sum":
      col_agg = [jnp.nansum(mat, axis=0) for mat in layer_matrices]
    else:
      raise ValueError(f"Unsupported aggregation: {aggr_op}")

    col_agg_clean = tree_map(lambda x: jnp.round(jnp.nan_to_num(x, nan=0.0), 3), col_agg)
    phi_shift = tree_map(lambda x: x[:, None], col_agg_clean)
    print(type(tree_unflatten(tree, scans)), type(phi_shift))

    return tree_map(lambda x, shift: fgi_dgaussian(x - 1.0 - phi_shift),
                    tree_unflatten(tree, scans), phi_shift)
  else:
    return tree_unflatten(tree, scans)


@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
      Args:
          q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
          q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
      Returns:
          new element ( A_out, Bu_out )
  """
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size // 4
    iA_ = A_i[0 * N: 1 * N]
    iB_ = A_i[1 * N: 2 * N]
    iC_ = A_i[2 * N: 3 * N]
    iD_ = A_i[3 * N: 4 * N]
    jA_ = A_j[0 * N: 1 * N]
    jB_ = A_j[1 * N: 2 * N]
    jC_ = A_j[2 * N: 3 * N]
    jD_ = A_j[3 * N: 4 * N]
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_
    Anew = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[0:N]
    b_i2 = b_i[N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = jnp.concatenate([new_b1, new_b2])

    return Anew, new_b + b_j


if __name__ == "__main__":
    # 1) Example

    # x = jnp.array([0,1,0,1,0,1,0,1,0], dtype=float)
    # print(f"input : {x}")
    # result= associative_scan(BO, x, apply_th=True, aggr_op="mean")
    # print(f"result(mean) : {result}, ")
    # result= associative_scan(BO, x, apply_th=True, aggr_op="sum")
    # print(f"result(sum) : {result}, ")
    # result= associative_scan(BO, x, apply_th=True, aggr_op="max")
    # print(f"result(max) : {result}, ")

    # 2) Example
    #P = 8  # Must be divisible by 4
    #seq_len = 5


    # Example data (replace with your own)
    #A_seq = jnp.stack([jnp.ones(P) for _ in range(seq_len)])  # shape: (seq_len, P)
    #b_seq = jnp.stack([jnp.zeros(P // 2) for _ in range(seq_len)])  # shape: (seq_len, P//2)

    #elems = (A_seq, b_seq)  # tuple of arrays

    # Use associative_scan
    #result = associative_scan(binary_operator, elems, apply_th=True)

    #print(result)
    def BO(x,y):
      return x+y
    x = jnp.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=float)
    re = associative_scan(BO, x, apply_th=True, aggr_op="mean")
    print(re)


