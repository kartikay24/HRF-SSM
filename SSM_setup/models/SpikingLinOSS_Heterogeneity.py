from typing import List, Callable, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from jax.nn.initializers import normal, zeros
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


def apply_hrf_im(A_diag, B, input_sequence, step, a1, a2, uv_learnable):
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

    if uv_learnable:
        F1 = M_IM_11 * Bu_elements * step + M_IM_11 * a1 + M_IM_12 * a2
        F2 = M_IM_21 * Bu_elements * step + M_IM_21 * a1 + M_IM_22 * a2
    else:
        F1 = M_IM_11 * Bu_elements * step
        F2 = M_IM_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IM_elements, F))
    ys = xs[:, A_diag.shape[0]:]
    return ys


def apply_hrf_imex(A_diag, B, input_sequence, step, a1, a2, uv_learnable):
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

    if uv_learnable:
        F1 = Bu_elements * step + A_ * a1 + B_ * a2
        F2 = Bu_elements * step + C_ * a1 + D_ * a2
    else:
        F1 = Bu_elements * step
        F2 = Bu_elements * (step ** 2.)
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IMEX_elements, F))
    # _, xs = associative_scan(binary_operator, (M_IMEX_elements, F), apply_th=True)
    ys = xs[:, A_diag.shape[0]:]
    return ys

def apply_hrf_imex_im_b(A_diag, G,  B, input_sequence, step, a1, a2, uv_learnable):
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

    if uv_learnable:
        F1 = Bu_elements * step + M_IM_11 * a1 + M_IM_12 * a2
        F2 = Bu_elements * step + M_IM_21 * a1 + M_IM_22 * a2
    else:
        F1 = M_IM_11 * Bu_elements * step
        F2 = M_IM_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IM_elements, F))
    ys = xs[:, A_diag.shape[0]:]
    return ys


class SpikingLinOSSLayer(eqx.Module):
    A_diag: jax.Array
    G: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array
    discretization: str
    heterogeneity_setup: str
    thresh_c: jax.Array
    thresh_d: jax.Array
    a1: jax.Array
    a2: jax.Array
    uv_learnable: bool

    def __init__(
            self,
            ssm_size,
            H,
            discretization,
            heterogeneity_setup,
            *,
            key
    ):

        B_key, C_key, D_key, A_key, step_key, C_thresh_key, D_thresh_key, G_key = jr.split(key, 8)
        self.A_diag = random.uniform(A_key, shape=(ssm_size,))
        # if 'b' in heterogeneity_setup:
        self.G = random.uniform(G_key, shape=(ssm_size,))  # G for IMEX
        self.B = simple_uniform_init(B_key, shape=(ssm_size, H, 2), std=1. / math.sqrt(H))
        self.C = simple_uniform_init(C_key, shape=(H, ssm_size, 2), std=1. / math.sqrt(ssm_size))
        self.D = normal(stddev=1.0)(D_key, (H,))
        self.steps = random.uniform(step_key, shape=(ssm_size,))
        self.thresh_c = random.uniform(C_thresh_key, shape=(ssm_size,), minval=0.0, maxval=1.0)
        self.thresh_d = random.uniform(D_thresh_key, shape=(H,), minval=0.0, maxval=1.0)
        self.discretization = discretization
        self.heterogeneity_setup = heterogeneity_setup
        if 'uv' in heterogeneity_setup:
            self.a1 = zeros(jr.key(0), shape=(ssm_size,))
            self.a2 = zeros(jr.key(1), shape=(ssm_size,))
            self.uv_learnable = True
        else:
            self.a1 = jnp.nan * jnp.ones((ssm_size,))
            self.a2 = jnp.nan * jnp.ones((ssm_size,))
            self.uv_learnable = False


    def __call__(self, input_sequence, spikes):
        # A_diag = nn.relu(self.A_diag)

        B_complex = self.B[..., 0]
        C_complex = self.C[..., 0]

        spikes += jnp.mean(input_sequence) * B_complex.shape[0] * B_complex.shape[1]  # scale by total number of elements to keep in line with other activations
        homogeneous_setup = self.heterogeneity_setup == 'homogeneous'
        if homogeneous_setup:
            ssm_dim = self.A_diag.shape[0]
            # Use constants in homogeneous mode so gradients do not flow through these parameters.
            A_diag = jnp.ones_like(self.A_diag) / ssm_dim
            steps = jnp.ones_like(self.steps) * 1e-3
            thresh_c = jnp.ones_like(self.thresh_c)
            thresh_d = jnp.ones_like(self.thresh_d)
        else:
            A_diag = nn.relu(self.A_diag)
            steps = nn.sigmoid(self.steps)
            thresh_c = self.thresh_c
            thresh_d = self.thresh_d

        if self.discretization == 'IMEX' and 'b' not in self.heterogeneity_setup:
            ys = apply_hrf_imex(A_diag, B_complex, input_sequence, steps, self.a1, self.a2, uv_learnable=self.uv_learnable)
        elif self.discretization == 'IM':
            ys = apply_hrf_im(A_diag, B_complex, input_sequence, steps, self.a1, self.a2, uv_learnable=self.uv_learnable)
        elif self.discretization == 'IMEX' and 'b' in self.heterogeneity_setup:
            G = nn.relu(self.G)
            LG = (G * steps + 2. - 2. * jnp.sqrt(steps * G + 1)) / (steps ** 2)
            UG = (G * steps + 2. + 2. * jnp.sqrt(steps * G + 1)) / (steps ** 2)
            A_diag = A_diag + nn.relu(A_diag - LG) - nn.relu(A_diag - UG)
            ys = apply_hrf_imex_im_b(A_diag, G, B_complex, input_sequence, steps, self.a1, self.a2, uv_learnable=self.uv_learnable)
        else:
            raise ValueError(f"Discretization type not implemented: {self.discretization}")
        ys = stepdoublegaussian(ys - thresh_c)
        spikes += jnp.mean(ys) * C_complex.shape[0]  * C_complex.shape[1]  # scale by total number of elements to keep in line with other activations 
        # print("th_C",jax.lax.stop_gradient(jnp.mean(ys)).val)
        ys = jax.vmap(lambda x: (C_complex @ x).real)(ys)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        output = ys + Du
        output = stepdoublegaussian(output - thresh_d)
        # spikes += jnp.sum(output)
        # print("th_D",jax.lax.stop_gradient(jnp.mean(output)).val)
        return output, spikes


class SpikingLinOSSBlock(eqx.Module):
    linear: eqx.nn.Linear
    ssm: SpikingLinOSSLayer
    drop: eqx.nn.Dropout
    norm: eqx.nn.BatchNorm
    thresh: jax.Array
    gsu: GSU
    heterogeneity_setup: str

    def __init__(
            self,
            ssm_size,
            H,
            discretization,
            drop_rate,
            heterogeneity_setup,
            *,
            key
    ):
        ssmkey, batchkey, linear_key, thresh_key = jr.split(key, 4)

        self.ssm = SpikingLinOSSLayer(
            ssm_size,
            H,
            discretization,
            heterogeneity_setup,
            key=ssmkey,
        )
        self.drop = eqx.nn.Dropout(p=drop_rate)
        self.linear = eqx.nn.Linear(H,H, key=linear_key)
        self.norm = eqx.nn.BatchNorm(
            input_size=H, axis_name="batch", channelwise_affine=False
        )
        # self.norm = eqx.nn.LayerNorm(shape=(H,))
        self.thresh = random.uniform(key=thresh_key, shape=(H,), minval=0.0, maxval=1.0)
        self.gsu = GSU(input_dim=H, output_dim=H, key=batchkey, alpha=0.05)
        self.heterogeneity_setup = heterogeneity_setup
        
    def __call__(self, x, state, spikes, *, key):
        """Compute SpikingLinOSS block."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, spikes = self.ssm(x, spikes)
        x = jnp.round(self.drop(x, key=dropkey1)) if self.drop.p > 0.00 else x
        # x = jax.vmap(self.gsu)(x)
        spikes += jnp.mean(x) * self.linear.weight.shape[0] * self.linear.weight.shape[1]  # scale by total number of elements to keep in line with other activations
        x = jax.vmap(self.linear)(x)
        x, state = self.norm(x.T, state)
        x = x.T
        # x = jax.vmap(self.norm)(x)
        x = stepdoublegaussian(x - self.thresh) if self.heterogeneity_setup!="homogeneous" else stepdoublegaussian(x - 1.0)
        # print("th",jax.lax.stop_gradient(jnp.mean(x)).val)
        x = jnp.round(self.drop(x, key=dropkey2)) if self.drop.p > 0.00 else x
        x = skip + x
        # spikes += jnp.sum(x)
        return x, state, spikes


class SpikingLinOSSwHeterogeneity(eqx.Module):
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
    heterogeneity_setup: str


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
            heterogeneity_setup,
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
                heterogeneity_setup=heterogeneity_setup,
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
        self.heterogeneity_setup = heterogeneity_setup


    def __call__(self, x, state, key):
        """Compute SpikingLinOSS."""
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)
        x, state = self.norm(x.T, state)
        x = x.T
        x = stepdoublegaussian(x - self.thresh) if self.heterogeneity_setup!="homogeneous" else stepdoublegaussian(x - 1.0)
        spikes = 0
        for block, key in zip(self.blocks, dropkeys):
            x, state, spikes = block(x, state, spikes, key=key)
        spikes += jnp.mean(x) * self.linear_layer.weight.shape[0] * self.linear_layer.weight.shape[1]  # scale by total number of elements to keep in line with other activations
        spikes = spikes * x.shape[0]  # scale by batch size to keep in line with other activations
        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = jax.nn.tanh(self.linear_layer(x))
            x = x[self.output_step - 1:: self.output_step]
        return x, state, spikes

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
    )[:, :, 0].T 
    return filtered