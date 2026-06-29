from typing import List, Callable, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from jax.nn.initializers import normal
import math
from jax import random
import sys
import os
from functional.autograd import stepdoublegaussian, fgi_dgaussian
from functional.scan import associative_scan

# Add the sibling directory to the Python path
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)


def simple_uniform_init(rng, shape, std=1.):
    weights = random.uniform(rng, shape) * 2. * std - std
    return weights


def ternary(x: jnp.ndarray, alpha: float = 0.05) -> jnp.ndarray:
    """Ternarize values based on dynamic threshold."""
    threshold = alpha * jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    return jnp.where(x >= threshold, 1.0,
                     jnp.where(x <= -threshold, -1.0, 0.0))


class GLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x):
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))

class GSU(eqx.Module):
    w1: eqx.nn.Linear
    alpha: float = eqx.static_field()

    def __init__(self, input_dim: int, output_dim: int, key: jax.Array, alpha: float = 0.05):
        self.alpha = alpha
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Gate 1: (Ter(x) @ W1 + b)
        x_ter = ternary(x, self.alpha)
        gate1 = self.w1(x_ter)

        # Gate 2: (x @ Ter(W1) + b)
        # We ternarize the weight manually, bypassing the Linear layer
        W2_ter = ternary(self.w1.weight, self.alpha)
        gate2 = jnp.dot(x, W2_ter.T) + self.w1.bias  # using W1.bias again for symmetry

        return gate1 * gate2


# Parallel scan operations
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


def apply_hrf_im(A_diag, B, input_sequence, step):
    """Compute the LxH output of LinOSS-IM given an LxH input.
    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        B       (complex64): input matrix            (P, H)
        C       (complex64): output matrix           (H, P)
        input_sequence (float32): input sequence of features    (L, H)
        step    (float):     discretization time-step $\Delta_t$  (P,)
    Returns:
        outputs (float32): the SSM outputs (LinOSS_IMEX layer preactivations)      (L, H)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    schur_comp = 1. / (1. + step ** 2. * A_diag)
    M_IM_11 = 1. - step ** 2. * A_diag * schur_comp
    M_IM_12 = -1. * step * A_diag * schur_comp
    M_IM_21 = step * schur_comp
    M_IM_22 = schur_comp

    M_IM = jnp.concatenate([M_IM_11, M_IM_12, M_IM_21, M_IM_22])

    M_IM_elements = M_IM * jnp.ones((input_sequence.shape[0],
                                     4 * A_diag.shape[0]))

    F1 = M_IM_11 * Bu_elements * step
    F2 = M_IM_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IM_elements, F))
    ys = xs[:, A_diag.shape[0]:]
    return ys


def apply_hrf_imex(A_diag, B, input_sequence, step):
    """Compute the LxH output of LinOSS-IMEX given an LxH input.
    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        B       (complex64): input matrix            (P, H)
        C       (complex64): output matrix           (H, P)
        input_sequence (float32): input sequence of features    (L, H)
        step    (float):     discretization time-step $\Delta_t$  (P,)
        threshold (float): threshold for the FGI-DGaussian activation function
    Returns:
        outputs (float32): the SSM outputs (LinOSS_IMEX layer preactivations)      (L, H)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    A_ = jnp.ones_like(A_diag)
    B_ = -1. * step * A_diag
    C_ = step
    D_ = 1. - (step ** 2.) * A_diag

    M_IMEX = jnp.concatenate([A_, B_, C_, D_])

    M_IMEX_elements = M_IMEX * jnp.ones((input_sequence.shape[0],
                                         4 * A_diag.shape[0]))

    F1 = Bu_elements * step
    F2 = Bu_elements * (step ** 2.)
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IMEX_elements, F))
    # _, xs = associative_scan(binary_operator, (M_IMEX_elements, F), apply_th=True)
    ys = xs[:, A_diag.shape[0]:]
    return ys

def apply_hrf_imex_im_b(A_diag, G,  B, input_sequence, step):
    """Compute the LxH output of LinOSS-IM given an LxH input.
    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        B       (complex64): input matrix            (P, H)
        C       (complex64): output matrix           (H, P)
        input_sequence (float32): input sequence of features    (L, H)
        step    (float):     discretization time-step $\Delta_t$  (P,)
    Returns:
        outputs (float32): the SSM outputs (LinOSS_IMEX layer preactivations)      (L, H)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    schur_comp = 1. / (1. + step * G)
    M_IM_11 = jnp.ones_like(A_diag) * schur_comp
    M_IM_12 = -1. * step * A_diag * schur_comp
    M_IM_21 = step * schur_comp
    M_IM_22 = (1. + (G * step)  - ((step ** 2.) * A_diag)) * schur_comp

    M_IM = jnp.concatenate([M_IM_11, M_IM_12, M_IM_21, M_IM_22])

    M_IM_elements = M_IM * jnp.ones((input_sequence.shape[0],
                                     4 * A_diag.shape[0]))

    F1 = M_IM_11 * Bu_elements * step
    F2 = M_IM_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IM_elements, F))
    ys = xs[:, A_diag.shape[0]:]
    return ys

# class BN_F(eqx.Module):
#     norm: eqx.nn.BatchNorm
#     u_cell: IFCell and LIFCell = eqx.field()
#     state_u: tuple[jnp.ndarray, jnp.ndarray]
#
#     def __init__(
#         self,
#         N: int,
#         H: int,
#         u_encoder: str,
#         *,
#         key: jax.random.PRNGKey
#     ):
#         coderkey, state_key = jax.random.split(key, 2)
#         self.norm = eqx.nn.BatchNorm(
#             input_size=N, axis_name="batch", channelwise_affine=False
#         )
#
#         if u_encoder == "LIF":
#             self.u_cell = LIFCell(N, H, key=coderkey)
#         elif u_encoder == "IF":
#             self.u_cell = IFCell(N, H, key=coderkey)
#         else:
#             raise NotImplementedError("LIF/IF not implemented")
#
#         z = jnp.zeros(shape=(100, N))
#         u = jax.random.uniform(key=state_key, shape=(100, H), minval=0.0, maxval=1.0)
#         self.state_u = z, u
#
#     def __call__(self, x, state):
#         x, state = self.norm(x.T, state)
#         x = x.T
#         vmapped_call = jax.vmap(self.u_cell)
#         x_out, state_u = vmapped_call(x, self.state_u)
#         return x_out, state_u, state


class SpikingLinOSSLayer(eqx.Module):
    A_diag: jax.Array
    G: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array
    discretization: str
    thresh_c: jax.Array
    thresh_d: jax.Array

    def __init__(
            self,
            ssm_size,
            H,
            discretization,
            *,
            key
    ):

        B_key, C_key, D_key, A_key, step_key, C_thresh_key, D_thresh_key, G_key = jr.split(key, 8)
        self.A_diag = random.uniform(A_key, shape=(ssm_size,))
        self.G = random.uniform(G_key, shape=(ssm_size,))  # G for IMEX
        self.B = simple_uniform_init(B_key, shape=(ssm_size, H, 2), std=1. / math.sqrt(H))
        self.C = simple_uniform_init(C_key, shape=(H, ssm_size, 2), std=1. / math.sqrt(ssm_size))
        self.D = normal(stddev=1.0)(D_key, (H,))
        self.steps = random.uniform(step_key, shape=(ssm_size,))
        self.thresh_c = random.uniform(C_thresh_key, shape=(ssm_size,), minval=0.0, maxval=1.0)
        self.thresh_d = random.uniform(D_thresh_key, shape=(H,), minval=0.0, maxval=1.0)
        self.discretization = discretization

    def __call__(self, input_sequence):
        # A_diag = nn.relu(self.A_diag)

        B_complex = self.B[..., 0]
        C_complex = self.C[..., 0]

        steps = nn.sigmoid(self.steps)
        if self.discretization == 'IMEX':
            A_diag = nn.relu(self.A_diag)
            ys = apply_hrf_imex(A_diag, B_complex, input_sequence, steps)
        elif self.discretization == 'IM':
            A_diag = nn.relu(self.A_diag)
            ys = apply_hrf_im(A_diag, B_complex, input_sequence, steps)
        elif self.discretization == 'IMEX_IM_B':
            G = nn.relu(self.G)
            LG = (G * self.steps + 2. - 2. * jnp.sqrt(self.steps*G+1))/(self.steps**2)
            UG = (G * self.steps + 2. + 2. * jnp.sqrt(self.steps*G+1))/(self.steps**2)
            A_diag = self.A_diag + nn.relu(self.A_diag - LG) - nn.relu(self.A_diag - UG)
            ys = apply_hrf_imex_im_b(A_diag, G, B_complex, input_sequence, steps)
        else:
            print('Discretization type not implemented')
        ys = stepdoublegaussian(ys - self.thresh_c)
        # print("th_C",jax.lax.stop_gradient(jnp.mean(ys)).val)
        ys = jax.vmap(lambda x: (C_complex @ x).real)(ys)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        output = ys + Du
        # output = stepdoublegaussian(output - self.thresh_d)
        # print("th_D",jax.lax.stop_gradient(jnp.mean(output)).val)
        return output, output.mean()


class SpikingLinOSSBlock(eqx.Module):
    # glu: GLU
    linear: eqx.nn.Linear
    ssm: SpikingLinOSSLayer
    drop: eqx.nn.Dropout
    norm: eqx.nn.LayerNorm
    thresh: jax.Array
    # gsu: GSU

    def __init__(
            self,
            ssm_size,
            H,
            discretization,
            drop_rate,
            *,
            key
    ):
        ssmkey, batchkey, linear_key, thresh_key = jr.split(key, 4)

        self.ssm = SpikingLinOSSLayer(
            ssm_size,
            H,
            discretization,
            key=ssmkey,
        )
        self.drop = eqx.nn.Dropout(p=drop_rate)
        # self.glu = GLU(H, H, key=linear_key)
        self.linear = eqx.nn.Linear(H,H, key=linear_key)
        # self.norm = eqx.nn.BatchNorm(
        #     input_size=H, axis_name="batch", channelwise_affine=False
        # )
        self.norm = eqx.nn.LayerNorm(shape=(H,))
        self.thresh = random.uniform(key=thresh_key, shape=(H,), minval=0.0, maxval=1.0)
        # self.gsu = GSU(input_dim=H, output_dim=H, key=batchkey, alpha=0.05)

    def __call__(self, x, state, *, key):
        """Compute SpikingLinOSS block."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, spikes = self.ssm(x)
        x = jnp.round(self.drop(x, key=dropkey1)) if self.drop.p > 0 else x
        # x = jax.vmap(self.gsu)(x)
        x = jax.vmap(self.linear)(x)
        # x = self.drop(jax.nn.gelu(x), key=dropkey1)
        # x = jax.vmap(self.glu)(x)
        # x, state = self.norm(x.T, state)
        # x = x.T
        # x = jax.vmap(self.norm)(x)
        x = stepdoublegaussian(x - self.thresh)
        # print("th",jax.lax.stop_gradient(jnp.mean(x)).val)
        x = jnp.round(self.drop(x, key=dropkey2)) if self.drop.p > 0 else x
        x = skip + x
        return x, state, spikes


class SpikingLinOSSwGeLU(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: List[SpikingLinOSSBlock]
    linear_layer: Union[eqx.nn.Linear,eqx.Module]
    norm: eqx.nn.BatchNorm
    classification: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False
    thresh: jax.Array


    def __init__(
            self,
            num_blocks,
            N,
            ssm_size,
            H,
            output_dim,
            classification,
            output_step,
            discretization,
            dropout,
            kernel,
            *,
            key
    ):

        linear_encoder_key, *block_keys, linear_layer_key, key_encoder_key = jr.split(
            key, num_blocks + 3
        )
        self.linear_encoder = eqx.nn.Linear(N, H, key=linear_encoder_key)
        self.norm = eqx.nn.BatchNorm(
            input_size=H, axis_name="batch", channelwise_affine=False
        )

        self.blocks = [
            SpikingLinOSSBlock(
                ssm_size,
                H,
                discretization=discretization,
                drop_rate=dropout,
                key=key,
            )
            for key in block_keys
        ]
        self.classification = classification
        self.output_step = output_step

        if self.classification:
            self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)
        else:
            self.linear_layer = LILinear(H, output_dim, key=linear_layer_key, kernel_size=kernel)
        self.thresh = random.uniform(linear_encoder_key, shape=(H,), minval=0.0, maxval=1.0)

    def delta_encode(self, x):
        dx = x[1:] - x[:-1]
        dx = jnp.concatenate([jnp.zeros_like(x[:1]), dx], axis=0)
        return dx

    def __call__(self, x, state, key):
        """Compute SpikingLinOSS."""
        dropkeys = jr.split(key, len(self.blocks))
        x = self.delta_encode(x)
        x = jax.vmap(self.linear_encoder)(x)
        # x, state = self.norm(x.T, state)
        # x = x.T
        x = stepdoublegaussian(x - self.thresh)
        # print("Encoder",jax.lax.stop_gradient(jnp.mean(x)).val)
        sop = 0
        for block, key in zip(self.blocks, dropkeys):
            x, state, spikes = block(x, state, key=key)
            sop += spikes
        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = jax.nn.tanh(self.linear_layer(x))
            x = x[self.output_step - 1:: self.output_step]
        return x, state

#Implementing convolution for efficient computation
class LILinear(eqx.Module):
  linear: eqx.nn.Linear
  kernel: jnp.ndarray # (kernel_size, 1, 1)

  def __init__(self, in_dim, out_dim, alpha=0.9, kernel_size=64, use_bias=True, key=None):
    linear_key, _ = jax.random.split(key)
    self.linear = eqx.nn.Linear(in_dim, out_dim, use_bias=use_bias, key=linear_key)
    decay = jnp.power(alpha, jnp.arange(kernel_size))
    self.kernel = ((1 - alpha) * decay)[:, None, None] # (K, 1, 1)


  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    # x: (L, H)
    x_proj = jax.vmap(self.linear)(x) # (L, out_dim)
    # Prepare for conv: (out_dim, T, 1)
    x_proj = x_proj.T[:, :, None]
    # Causal convolution
    filtered = jax.lax.conv_general_dilated(
      x_proj,
      self.kernel,
      window_strides=(1,),
      padding=[(self.kernel.shape[0] - 1, 0)],
      dimension_numbers=("NWC", "WIO", "NWC"),
    )[:, :, 0].T # (L, out_dim)
    return filtered
