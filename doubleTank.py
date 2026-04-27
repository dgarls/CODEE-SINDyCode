import time

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp

volume_A = 5
volume_B = 5

numPoints = 4000
t_span = (0, 10)
t_eval = np.linspace(*t_span, numPoints)
dt = (t_span[1] - t_span[0]) / numPoints
initial_state = [1, 3]

np.random.seed(459)


def noisy_flow(base_flow, noise_strength=0.1):
    return base_flow * (1 + noise_strength * np.random.randn())


def concentration_in_A(t):
    return 1


def flow_in_A(t):
    return noisy_flow(1)


def flow_out_A_in_B(t):
    return 2


def flow_out_B_in_A(t):
    return 1


def flow_out_B(t):
    return 1


def tank(t, X):
    A, B = X

    net_flow_A = flow_in_A(t) - flow_out_A_in_B(t) + flow_out_B_in_A(t)
    net_flow_B = flow_out_A_in_B(t) - flow_out_B(t) - flow_out_B_in_A(t)

    pipe_in_A = concentration_in_A(t) * flow_in_A(t)
    pipe_from_B_to_A = (B / (volume_B + net_flow_B)) * flow_out_B_in_A(t)
    pipe_from_A_to_B = (A / (volume_A + net_flow_A)) * flow_out_A_in_B(t)
    pipe_out_B = flow_out_B(t) * (B / (volume_B + net_flow_B))

    return np.array(
        [
            pipe_in_A + pipe_from_B_to_A - pipe_from_A_to_B,
            -pipe_out_B - pipe_from_B_to_A + pipe_from_A_to_B,
        ]
    )


sol = solve_ivp(tank, t_span=t_span, t_eval=t_eval, y0=initial_state)
X = sol.y.T

plt.figure()
plt.plot(t_eval, X[:, 0], label="A")
plt.plot(t_eval, X[:, 1], label="B")
plt.grid()
plt.legend()
plt.show()


lib = ps.PolynomialLibrary(degree=1, include_bias=True)


model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.01),
    feature_library=lib,
    feature_names=["A", "B"],
)


model.fit(x=X, t=t_eval)
model.print()
sim = model.simulate(t=t_eval, x0=initial_state)


plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t_eval, X[:, 0], ".", label="A Real", markersize=2)
plt.plot(t_eval, sim[:, 0], "-", label="A SINDy")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_eval, X[:, 1], ".", label="B Real", color="purple", markersize=2)
plt.plot(t_eval, sim[:, 1], "-", label="B SINDy", color="red")
plt.legend()

plt.grid()
plt.tight_layout()
plt.show()

# ----------- R^2 calculation -----------


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


r2_A = r2_score(X[:, 0], sim[:, 0])
r2_B = r2_score(X[:, 1], sim[:, 1])

print(f"R^2 for A: {r2_A:.5f}")
print(f"R^2 for B: {r2_B:.5f}")
