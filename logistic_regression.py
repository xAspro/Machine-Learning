import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import chi2

dose = np.array([0.1, 0.3, 0.4, 0.5, 0.55, 0.7, 0.8, 1.0])
efficiency = np.array([False, False, False, True, False, True, True, True])

print("----- Initial Data -----")
print("Dose")
for i in range(len(dose)):
    print(dose[i])

print("\nEfficiency")
for i in range(len(efficiency)):
    print(efficiency[i].astype(int))

print("------------------------\n")

def plot_data(dose, efficiency, title, fitted_curve=None):
    plt.scatter(dose, efficiency, c=efficiency, cmap='coolwarm_r')
    if fitted_curve is not None:
        dose_range = np.linspace(min(dose), max(dose), 100)
        plt.plot(dose_range, fitted_curve(dose_range), color='black', label='Fitted Curve')
        plt.legend()
    plt.colorbar(label='Efficiency')
    plt.xlabel('Dose')
    plt.ylabel('Efficiency')
    plt.title(title)
    plt.show()

plot_data(dose, efficiency, title='Initial Data')

# Sigmoid function y = L + (U - L) / (1 + exp(-k*(x - x0)))
# But since this is for probability, we want L=0 and U=1, we simplify to:
# y = 1 / (1 + exp(-k*(x - x0)))
# def restricted_sigmoid(x, params):
#     k, x0 = params
#     y = 1 / (1 + np.exp(-k * (x - x0)))
#     return y

def logit(p):
    return np.log(p / (1 - p))

def inverse_logit(logit_value):
    return 1 / (1 + np.exp(-logit_value))

def log_likelihood(params, dose, efficiency):
    k, x0 = params
    logit_eff = k * (dose - x0)
    prob_eff = inverse_logit(logit_eff)
    ll = np.sum(efficiency * np.log(prob_eff + 1e-12) + (1 - efficiency) * np.log(1 - prob_eff + 1e-12))
    return ll

# Transforming the data for better fitting
# logit_efficiency = logit(efficiency.astype(float) + 1e-12)  # add small value to avoid log(0)

# fitted_curve = partial(restricted_sigmoid, params=(100, 0.5))
# plot_data(dose, efficiency, fitted_curve=fitted_curve)

# McFadden R² calculation
def r2_score(params, efficiency, dose, ll_model, ll_null):
    r2 = 1 - (ll_model / ll_null)
    return r2

def p_value(params, dose, efficiency, ll_model, ll_null):
    lr_stat = 2 * (ll_model - ll_null)
    p_val = 1 - chi2.cdf(lr_stat, df=1)
    return p_val

def calculate_ll(params_1, dose, efficiency):
    params_2 = (0, 0)  # Null model with no predictors
    ll_model = log_likelihood(params_1, dose, efficiency)
    ll_null = log_likelihood(params_2, dose, efficiency)
    return ll_model, ll_null

def gradient_descent(params, dose, efficiency, learning_rate, num_iterations, tol=1e-12):
    k, x0 = params
    prev_cost = log_likelihood((k, x0), dose, efficiency)
    for _ in range(num_iterations):
        logit_eff = k * (dose - x0)
        prob_eff = inverse_logit(logit_eff)

        error = efficiency - prob_eff

        dk = np.sum(error * (dose - x0))
        dx0 = -np.sum(error * k)

        k += learning_rate * dk
        x0 += learning_rate * dx0

        current_cost = log_likelihood((k, x0), dose, efficiency)
        # print(f"Current Log-Likelihood: {current_cost}")
        # print(f"Parameters: k={k}, x0={x0}")
        # print("prev_cost:", prev_cost, "current_cost:", current_cost)
        # print("current_cost - prev_cost:", current_cost - prev_cost)
        # print("-----\n")
        if abs(current_cost - prev_cost) < tol:
            print("\t\t\t\tConvergence reached.")
            break

        prev_cost = current_cost

        # if _ == 10:
        #     break

    return k, x0

def run_regression(dose, efficiency, learning_rate=0.01, num_iterations=10000):
    initial_params = (1, 0.35)  # Initial guess for k and x0
    optimized_params = gradient_descent(initial_params, dose, efficiency, learning_rate, num_iterations)
    ll_model, ll_null = calculate_ll(optimized_params, dose, efficiency)

    cost = log_likelihood(optimized_params, dose, efficiency)
    r2 = r2_score(optimized_params, efficiency, dose, ll_model, ll_null)
    p_val = p_value(optimized_params, dose, efficiency, ll_model, ll_null)
    print("\n----- Optimization Results -----")
    print("Function: restricted sigmoid function at z = k*(x - x0)")
    print(f"Optimized Parameters: k={optimized_params[0]}, x0={optimized_params[1]}")
    print(f"Final Log-Likelihood: {cost}")
    print(f"McFadden R²: {r2}")
    print(f"P-value: {p_val}")
    print("---------------------------------\n")

    return optimized_params

def main():
    initial_params = (1.0, 0.3)  # Initial guess for k and x0
    ll_model, ll_null = calculate_ll(initial_params, dose, efficiency)
    r2 = r2_score(initial_params, efficiency, dose, ll_model, ll_null)
    p_val = p_value(initial_params, dose, efficiency, ll_model, ll_null)

    optimal_params = run_regression(dose, efficiency, learning_rate=0.01, num_iterations=500000)

    def fitted_curve(x, params=initial_params):
        return inverse_logit(params[0] * (x - params[1]))
    
    plot_data(dose, efficiency, title='Initial Guess', fitted_curve=fitted_curve)
    plot_data(dose, efficiency, title='Optimized Parameters', fitted_curve=partial(fitted_curve, params=optimal_params))


if __name__ == "__main__":
    main()




