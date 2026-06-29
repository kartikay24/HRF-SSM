import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import lax

#########################BASE########################################################

@jax.jit
def step(x: jnp.ndarray) -> jnp.ndarray:
    return (x > 0.0).astype(jnp.float32)


@jax.jit
def gaussian(x: jnp.ndarray, mu: float = 0.0, sigma: float = 1.0) -> jnp.ndarray:
  return (1.0 / (sigma * jnp.sqrt(2 * jnp.pi))) * jnp.exp(-((x - mu) ** 2) / (2.0 * (sigma ** 2)))


@jax.jit
def DoubleGaussian(x: jnp.ndarray) -> jnp.ndarray:
    p = 0.15
    scale = 6.0
    length = 0.5
    gamma = 0.5

    sigma1 = length
    sigma2 = scale * length

    return (gamma * (1.0 + p) * gaussian(x, mu=0.0, sigma=sigma1)
            - p * gaussian(x, mu=length, sigma=sigma2)
            - p * gaussian(x, mu=-length, sigma=sigma2))

#########################BASE########################################################

# step gaussian
@jax.custom_vjp
def stepgaussian(x: jnp.ndarray) -> jnp.ndarray:
    return step(x)

def stepgaussian_fwd(x: jnp.ndarray):
    return step(x), x

def stepgaussian_bwd(res, g):
    x = res
    dfdx = gaussian(x)
    return (g * dfdx,)

stepgaussian.defvjp(stepgaussian_fwd, stepgaussian_bwd)


# step linear
@jax.custom_vjp
def steplinear(x: jnp.ndarray) -> jnp.ndarray:
    return step(x)

def steplinear_fwd(x: jnp.ndarray):
    return step(x), x

def steplinear_bwd(res, g):
    x = res
    dfdx = jnp.maximum(0.0, 1.0 - jnp.abs(x))
    return (g * dfdx,)

steplinear.defvjp(steplinear_fwd, steplinear_bwd)


# step exp
@jax.custom_vjp
def stepexp(x: jnp.ndarray) -> jnp.ndarray:
    return step(x)

def stepexp_fwd(x: jnp.ndarray):
    return step(x), x

def stepexp_bwd(res, g):
    x = res
    dfdx = jnp.exp(-jnp.abs(x))
    return (g * dfdx,)

stepexp.defvjp(stepexp_fwd, stepexp_bwd)


# step multi gaussian
@jax.custom_vjp
def stepmultigaussian(x: jnp.ndarray) -> jnp.ndarray:
    return step(x)

def stepmultigaussian_fwd(x: jnp.ndarray):
    return step(x), x

def stepmultigaussian_bwd(res, g):
    x = res
    p = 0.15
    scale = 6.0
    length = 0.5
    sigma1 = length
    sigma2 = scale * length
    gamma = 0.5

    dfdx = (1. + p) * gaussian(x, mu=0., sigma=sigma1) \
           - p * gaussian(x, mu=length, sigma=sigma2) \
           - p * gaussian(x, mu=-length, sigma=sigma2)

    return (g * dfdx * gamma,)

stepmultigaussian.defvjp(stepmultigaussian_fwd, stepmultigaussian_bwd)


# step dpuble gaussian
@jax.custom_vjp
def stepdoublegaussian(x: jnp.ndarray) -> jnp.ndarray:
    return step(x)

def stepdoublegaussian_fwd(x: jnp.ndarray):
    return step(x), x

def stepdoublegaussian_bwd(res, g):
    x = res
    p = 0.15
    scale = 6.0
    length = 0.5
    sigma1 = length
    sigma2 = scale * length
    gamma = 0.5

    dfdx = (1. + p) * gaussian(x, mu=0., sigma=sigma1) - 2. * p * gaussian(x, mu=0., sigma=sigma2)
    return (g * dfdx * gamma,)

stepdoublegaussian.defvjp(stepdoublegaussian_fwd, stepdoublegaussian_bwd)


# fgi_dgaussian
def fgi_dgaussian(x: jnp.ndarray) -> jnp.ndarray:
    x_detached = lax.stop_gradient(step(x))

    p = 0.15
    scale = 6.0
    length = 0.5
    sigma1 = length
    sigma2 = scale * length
    gamma = 0.5

    df = (1. + p) * gaussian(x, mu=0., sigma=sigma1) - 2. * p * gaussian(x, mu=0., sigma=sigma2)
    df_detached = lax.stop_gradient(df)

    dfd = gamma * df_detached * x
    dfd_detached = lax.stop_gradient(dfd)

    return dfd - dfd_detached + x_detached



if __name__ == "__main__": 
    
    # Checking the working of the surrogate function.
    import jax
    import jax.numpy as jnp
    import optax
    import equinox as eqx
    from jax import random

    # model
    class SpikingLayer(eqx.Module):
        weight: jnp.ndarray
        bias: jnp.ndarray
        threshold: float = 1.0

        def __init__(self, in_dim, out_dim, key):
            wkey, bkey = jax.random.split(key)
            self.weight = jax.random.normal(wkey, (out_dim, in_dim)) * 0.1
            self.bias = jax.random.normal(bkey, (out_dim,)) * 0.01

        def __call__(self, x):
            # print("SpikingLayer input x shape:", x.shape)  # Expect (batch, in_dim)
            u = jnp.dot(x, self.weight.T) + self.bias       # (batch, out_dim)
            # print("SpikingLayer membrane potential u shape:", u.shape)
            spikes = stepdoublegaussian(u - self.threshold)
            # spikes = fgi_dgaussian(u - self.threshold)
            # print("SpikingLayer output (spikes) shape:", spikes.shape)
            return spikes
            
    class SimpleSNN(eqx.Module):
        layer1: SpikingLayer
        layer2: eqx.nn.Linear

        def __init__(self, in_dim, hidden_dim, out_dim, key):
            key1, key2 = jax.random.split(key)
            self.layer1 = SpikingLayer(in_dim, hidden_dim, key1)
            self.layer2 = eqx.nn.Linear(hidden_dim, out_dim, key=key2)

        def __call__(self, x):
            num_spikes = 0
            # print("Model input x shape:", x.shape)
            spikes = self.layer1(x)  # should be (batch, hidden_dim)
            # print("Spikes from layer1:", spikes.shape)
            out = self.layer2(spikes.T)  # should be (batch, out_dim)
            # print("Final output shape:", out.shape)
            num_spikes = jnp.sum(spikes)
            return out, num_spikes
    
    # # loss function
    # def mse_loss(model, x, y):
    #     # print("x shape:", x.shape)
    #     preds = model(x)
    #     # print("preds shape:", preds.shape)
    #     # print("y shape:", y.shape)
    #     return jnp.mean((preds - y) ** 2)

    # train_step
    def make_train_step(optimizer):
        # @jax.jit
        # def train_step(model, opt_state, x, y):
        #     (loss, num_spikes), grads = jax.value_and_grad(mse_loss)(model, x, y)
        #     updates, new_opt_state = optimizer.update(grads, opt_state)
        #     model = eqx.apply_updates(model, updates)
        #     return model, new_opt_state, loss
        # return train_step
        def train_step(model, opt_state, x, y):
            def loss_fn(model, x, y):
                preds, num_spikes = model(x)
                loss = jnp.mean((preds - y) ** 2)
                return loss, num_spikes

            (loss, num_spikes), grads = jax.value_and_grad(loss_fn, has_aux=True)(model, x, y)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, new_opt_state, loss, num_spikes
        return train_step


    # loop
    def train():
        key = random.PRNGKey(0)
        model = SimpleSNN(in_dim=10, hidden_dim=20, out_dim=1, key=key)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(model)

        # Dummy data
        x = random.normal(key, (32, 10))  # batch of 32 inputs
        y = jnp.ones((32, 1), dtype=jnp.float32)             # dummy target
        train_step = make_train_step(optimizer)

        for epoch in range(100):
            # model, opt_state, loss = train_step(model, opt_state, x, y)
            model, opt_state, loss, num_spikes = train_step(model, opt_state, x, y)
            if epoch % 10 == 0:
                # print(f"Epoch {epoch}: Loss = {loss:.4f}")
                print(f"Epoch {epoch}: Loss = {loss:.4f}, SOPs = {num_spikes/y.size}")

    train()