import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp

"""
This is a demonstration of how SINDy works with our toy
problem with xdot = x.
"""


# We define our ODE as a function for solve_ivp
def ode(t, y):
    return y


endTime = 3
tspan = (0, endTime)
dt = 1
t = np.arange(
    0, endTime, dt
)  # Our time array, all the points we will evaluate our ODE at

print(t)

x0 = [1]  # Our initial state at time 0, which we define to be 1

data = solve_ivp(
    ode, t_span=tspan, t_eval=t, y0=x0
)  # These are our simulated data points

X = data.y.T  # Unpacking the solution

# SINDy Model Creation
lib = ps.PolynomialLibrary(degree=2, include_bias=False)

model = ps.SINDy(
    feature_library=lib, optimizer=ps.STLSQ(threshold=1e-3), feature_names=["X"]
)

model.fit(x=X, t=t)

model.print()  # Output: (X)' = 1.124 X + -0.039 X^2
# This is very similar to what we found by hand!
# It is slightly improved by STLSQ.

# Let's try increasing the density of data points to get a more accurate answer.

dt = 0.5
t = np.arange(0, endTime, dt)
data = solve_ivp(ode, t_span=tspan, t_eval=t, y0=x0)
X = data.y.T
model.fit(x=X, t=t)
model.print()  # Ever so slightly better... let's try decreasing dt again.

dt = 0.01
t = np.arange(0, endTime, dt)
data = solve_ivp(ode, t_span=tspan, t_eval=t, y0=x0)
X = data.y.T
model.fit(x=X, t=t)
model.print()  # Boom! There we go.
# If our simulated data points are dense enough, we get an accurate
# SINDy prediction for the original ODE. This is pretty epic.
