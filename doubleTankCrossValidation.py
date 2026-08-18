import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp

volume_A = 5
volume_B = 5

numPoints = 8000
t_span = (0, 10)
t_eval = np.linspace(*t_span, numPoints)
dt = (t_span[1] - t_span[0]) / numPoints
initial_state = [1, 3]

np.random.seed(459)

trueCoeffs = [
    [1, -0.4, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0.4, -0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]


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
    current_flow_in_A = flow_in_A(t)

    net_flow_A = current_flow_in_A - flow_out_A_in_B(t) + flow_out_B_in_A(t)
    net_flow_B = flow_out_A_in_B(t) - flow_out_B(t) - flow_out_B_in_A(t)

    pipe_in_A = concentration_in_A(t) * current_flow_in_A
    pipe_from_B_to_A = (B / (volume_B + net_flow_B)) * flow_out_B_in_A(t)
    pipe_from_A_to_B = (A / (volume_A + net_flow_A)) * flow_out_A_in_B(t)
    pipe_out_B = (B / (volume_B + net_flow_B)) * flow_out_B(t)

    return np.array(
        [
            pipe_in_A + pipe_from_B_to_A - pipe_from_A_to_B,
            -pipe_out_B - pipe_from_B_to_A + pipe_from_A_to_B,
        ]
    )


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# Training trajectory
sol = solve_ivp(tank, t_span=t_span, t_eval=t_eval, y0=initial_state)
X = sol.y.T

testICs = [[4, 2]]#, [2, 4], [3, 1], [0.5, 4.5]] # Can add more CV ICs
testTrajs = []
for ic in testICs:
    traj = solve_ivp(tank, t_span=t_span, t_eval=t_eval, y0=ic)
    testTrajs.append(traj.y.T)

#plt.figure()
#plt.plot(t_eval, X[:, 0], label="A")
#plt.plot(t_eval, X[:, 1], label="B")
#plt.grid()
#plt.legend()
#plt.show()

plt.figure(dpi=600)
plt.plot(t_eval, testTrajs[0][:, 0], label="A")
plt.plot(t_eval, testTrajs[0][:, 1], label="B")
plt.grid()
plt.legend()
plt.savefig("doubleTankCVTraj.png")
#plt.show()

lib = ps.PolynomialLibrary(degree=4, include_bias=True)

alphaVals = np.linspace(0, 1, 20)
thresholdVals = np.linspace(0, 1, 20)

results = []  
              
for alpha in alphaVals:
    for threshold in thresholdVals:
        print(f"Alpha Val: {alpha}")
        model = ps.SINDy(
            optimizer=ps.STLSQ(
                threshold=threshold,
                alpha=alpha,
                unbias=False,
                normalize_columns=False,
                max_iter=100,
            ),
            feature_library=lib,
            differentiation_method=ps.SmoothedFiniteDifference(),
        )
        model.fit(x=X, t=t_eval, feature_names=["A", "B"])

        coeffs = model.coefficients()
        n_terms = int(np.sum(coeffs != 0))

        entry = {
            "alpha": alpha,
            "threshold": threshold,
            "coeffs": coeffs,
            "equations": model.equations(),
            "n_terms": n_terms,
            "r2_A": np.nan,
            "r2_B": np.nan,
            "r2_total": np.nan,
            "coeff_error": np.nan,
            "cv_error": -np.inf,
        }

        if n_terms == 0:
            print("Model with all 0s")
            results.append(entry)
            continue

        model.print()

        try:
            sim = model.simulate(t=t_eval, x0=initial_state)
            r2_A = r2_score(X[:, 0], sim[:, 0])
            r2_B = r2_score(X[:, 1], sim[:, 1])
            entry["r2_A"] = r2_A
            entry["r2_B"] = r2_B
            entry["r2_total"] = r2_A + r2_B
            print(f"R^2 for A: {r2_A:.5f}")
            print(f"R^2 for B: {r2_B:.5f}")
        except Exception:
            print("Model could not be simulated from initial_state.")

        # Distance to true coefficients
        coeffErrA = np.linalg.norm(coeffs[0] - trueCoeffs[0])
        coeffErrB = np.linalg.norm(coeffs[1] - trueCoeffs[1])
        entry["coeff_error"] = coeffErrA + coeffErrB
        print(f"Coefficient Error: {entry['coeff_error']}\n")

        # Cross-validation
        cv_errors = []
        for ic, testX in zip(testICs, testTrajs):
            try:
                sim2 = model.simulate(t=t_eval, x0=ic)
                cv_errors.append(
                    r2_score(testX[:, 0], sim2[:, 0]) + r2_score(testX[:, 1], sim2[:, 1])
                )
            except Exception:
                cv_errors.append(-np.inf)

        entry["cv_error"] = float(np.mean(cv_errors))
        print(f"Cross Validate Error: {entry['cv_error']}")

        results.append(entry)

# Selection
finite_results = [r for r in results if np.isfinite(r["cv_error"])]

if not finite_results:
    raise RuntimeError("No model produced a finite CV error across the grid.")

best_by_coeff_error = min(
    results, key=lambda r: r["coeff_error"] if np.isfinite(r["coeff_error"]) else np.inf
)
best_by_cv = max(finite_results, key=lambda r: r["cv_error"])

print(f"Best Total Coefficient Error: {best_by_coeff_error['coeff_error']}")
print(f"Best alpha, threshold: {best_by_coeff_error['alpha']}, {best_by_coeff_error['threshold']}")
print(f"Best coefficients for both: {best_by_coeff_error['coeffs']}")
print(f"R2 for best coefficients: {best_by_coeff_error['r2_total']}")

r2_all = sorted([r["r2_total"] for r in results if np.isfinite(r["r2_total"])], reverse=True)
print(f"Best Coefficients have rank {r2_all.index(best_by_coeff_error['r2_total'])}")

print(f"\nBest Cross-Validation Error: {best_by_cv['cv_error']}")

# Top 10 by CV error
top10 = sorted(finite_results, key=lambda r: r["cv_error"], reverse=True)[:10]

print("\nTop 10 CV Errors:")
for i, r in enumerate(top10, start=1):
    print(f"\nModel {i}:")
    print(f"alpha={r['alpha']}, threshold={r['threshold']}, cv_error={r['cv_error']}")
    print(r["coeffs"])

print("\n FINAL MODEL:")
print(f"Alpha: {best_by_cv['alpha']}")
print(f"Threshold: {best_by_cv['threshold']}")
print(f"R2 for A: {best_by_cv['r2_A']}")
print(f"R2 for B: {best_by_cv['r2_B']}")
print(best_by_cv["coeffs"])

# Plot true vs predicted trajectories for the best model
best_model = ps.SINDy(
    optimizer=ps.STLSQ(
        threshold=best_by_cv["threshold"],
        alpha=best_by_cv["alpha"],
        unbias=False,
        normalize_columns=False,
        max_iter=100,
    ),
    feature_library=lib,
    differentiation_method=ps.SmoothedFiniteDifference(),
)
best_model.fit(x=X, t=t_eval, feature_names=["A", "B"])
sim_best = best_model.simulate(t=t_eval, x0=initial_state)

plt.figure(dpi=600)
plt.subplot(2, 1, 1)
plt.plot(t_eval, X[:, 0], ".", label="A Real", markersize=2)
plt.plot(t_eval, sim_best[:, 0], "-", label="A SINDy")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_eval, X[:, 1], ".", label="B Real", color="purple", markersize=2)
plt.plot(t_eval, sim_best[:, 1], "-", label="B SINDy", color="red")
plt.legend()

plt.grid()
plt.tight_layout()
plt.savefig("doubleTank4th.png")
plt.show()

# Cross-validation graphs
sim_cv = best_model.simulate(t=t_eval, x0=testICs[0])
plt.figure(dpi=600)
plt.subplot(2, 1, 1)
plt.plot(t_eval, testTrajs[0][:, 0], ".", label="A Real", markersize=2)
plt.plot(t_eval, sim_cv[:, 0], "-", label="A SINDy")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_eval, testTrajs[0][:, 1], ".", label="B Real", color="purple", markersize=2)
plt.plot(t_eval, sim_cv[:, 1], "-", label="B SINDy", color="red")
plt.legend()

plt.grid()
plt.tight_layout()
plt.savefig("doubleTankCV.png")
plt.show()