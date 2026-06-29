import torch
from snn import functional
from .linear_layer import LinearMask
################################################################
# Neuron update functional
################################################################

DEFAULT_MASK_PROB = 0

TRAIN_B_offset = True
DEFAULT_RF_B_offset = 1.

# here: Depends on the initialization
DEFAULT_RF_ADAPTIVE_B_offset_a = 1
DEFAULT_RF_ADAPTIVE_B_offset_b = 6

TRAIN_OMEGA = True
DEFAULT_RF_OMEGA = 10.

DEFAULT_RF_ADAPTIVE_OMEGA_a = 10
DEFAULT_RF_ADAPTIVE_OMEGA_b = 50

DEFAULT_RF_THETA = 1 # .99

# Reset: Keep (1 - Zeta) of the membrane potential
# start with constant initialization
TRAIN_ZETA = False
DEFAULT_RF_ZETA = .00

DEFAULT_RF_ADAPTIVE_ZETA_a = 0
DEFAULT_RF_ADAPTIVE_ZETA_b = 0

TRAIN_DT = False
DEFAULT_DT = 0.01
DEFAULT_RF_ADAPTIVE_DT = 0.01


def hrf_update(
        x: torch.Tensor,  # injected current: input x weight
        u: torch.Tensor,  # membrane potential (complex value)
        v: torch.Tensor,
        b: torch.Tensor,  # attraction to resting state
        omega: torch.Tensor,  # eigen ang. frequency of the neuron
        dt: float = DEFAULT_DT,  # torch.Tensor 0.01
        theta: float = DEFAULT_RF_THETA,
        scheme: str = 'imex2',  # 'euler forward' or 'rk4'
):

    # damped oscillatory activity dim = (1, hidden_size)
    # membrane update u dim = (batch_size, hidden_size)
    if scheme == 'euler forward':
        v1 = v + u.mul(dt)
        u1 = u + x.mul(dt)-b.mul(u).mul(2*dt)-torch.square(omega).mul(v).mul(dt)
        u = u1
        v = v1
    
    elif scheme == 'imex':  # IMEX_IM_B
        u = (u + x.mul(dt)-torch.square(omega).mul(v).mul(dt))/(1 + b.mul(2*dt))
        v = v + u.mul(dt)

    elif scheme == 'imex2':   # IMEX_EX_B
        u = u + x.mul(dt)-b.mul(u).mul(2*dt)-torch.square(omega).mul(v).mul(dt)
        v = v + u.mul(dt)

    elif scheme == 'euler backward':
        a = 1 + b.mul(2*dt)
        denom = a + torch.square(omega.mul(dt))
        v1 = (u.mul(dt) + v + b.mul(2*dt).mul(v) + x.mul(dt**2))/denom
        u1 = (v1 - v)/dt
        u = u1
        v = v1

    # generate spike
    z = functional.StepDoubleGaussianGrad.apply(u - theta)

    # reset membrane potential
    # u = u.mul(1 - z.mul(theta).mul(zeta))
    # v = v.mul(1 - z.mul(theta).mul(zeta))
    return z, u, v


################################################################
# Layer classes
################################################################
class HRFCell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            layer_size: int,
            mask_prob: float = DEFAULT_MASK_PROB,
            b_offset: float = DEFAULT_RF_B_offset,
            adaptive_b_offset: bool = TRAIN_B_offset,
            adaptive_b_offset_a: float = DEFAULT_RF_ADAPTIVE_B_offset_a,
            adaptive_b_offset_b: float = DEFAULT_RF_ADAPTIVE_B_offset_b,
            omega: float = DEFAULT_RF_OMEGA,
            adaptive_omega: bool = TRAIN_OMEGA,
            adaptive_omega_a: float = DEFAULT_RF_ADAPTIVE_OMEGA_a,
            adaptive_omega_b: float = DEFAULT_RF_ADAPTIVE_OMEGA_b,
            dt: float = DEFAULT_DT,
            bias: bool = False,
            scheme: str = 'imex2', # 'euler forward', 'euler backward', 'imex', 'imex2'
    ) -> None:
        super(HRFCell, self).__init__()

        self.input_size = input_size
        self.layer_size = layer_size
        self.scheme = scheme

        # LinearMask: applies mask only to hidden recurrent weights in forward pass
        # linear.weight initialized with xavier_uniform_
        # self.mask_prob = mask_prob
        #
        # self.linear = rf.LinearMask(
        #     in_features=input_size,
        #     out_features=layer_size,
        #     bias=bias,
        #     mask_prob=mask_prob,
        #     lbd=input_size - layer_size,
        #     ubd=input_size,
        # )

        self.linear = torch.nn.Linear(
            in_features=input_size,
            out_features=layer_size,
            bias=bias
        )

        torch.nn.init.xavier_uniform_(self.linear.weight)

        self.adaptive_omega = adaptive_omega
        self.adaptive_omega_a = adaptive_omega_a
        self.adaptive_omega_b = adaptive_omega_b

        omega = omega * torch.ones(layer_size)

        if adaptive_omega:
            self.omega = torch.nn.Parameter(omega)
            torch.nn.init.uniform_(self.omega, adaptive_omega_a, adaptive_omega_b)
        else:
            self.register_buffer('omega', omega)


        self.adaptive_b_offset = adaptive_b_offset
        self.adaptive_b_a = adaptive_b_offset_a
        self.adaptive_b_b = adaptive_b_offset_b

        b_offset = b_offset * torch.ones(layer_size)

        if adaptive_b_offset:
            self.b_offset = torch.nn.Parameter(b_offset)
            torch.nn.init.uniform_(self.b_offset, adaptive_b_offset_a, adaptive_b_offset_b)
        else:
            self.register_buffer('b_offset', b_offset)

        self.dt = dt

    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        in_sum = self.linear(x)

        z, u, v = state

        omega = torch.abs(self.omega)

        b = -torch.abs(self.b_offset)

        z, u, v = hrf_update(
            x=in_sum,
            u=u,
            v=v,
            b=b,
            omega=omega,
            dt=self.dt,
            scheme=self.scheme,
        )

        return z, u, v
