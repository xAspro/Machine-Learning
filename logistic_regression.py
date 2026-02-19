import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import chi2

dose = np.array([0.1, 0.3, 0.4, 0.5, 0.55, 0.7, 0.8, 1.0])
efficiency = np.array([False, False, False, True, False, True, True, True])

# # For checking against Logistic Regression Calculator
# print("----- Initial Data -----")
# print("Dose")
# for i in range(len(dose)):
#     print(dose[i])

# print("\nEfficiency")
# for i in range(len(efficiency)):
#     print(efficiency[i].astype(int))

# print("------------------------\n")

def plot_data(dose, efficiency, title, fitted_curves=None):
    plt.scatter(dose, efficiency, c=efficiency, cmap='coolwarm_r')
    if fitted_curves is not None:
        dose_range = np.linspace(min(dose), max(dose), 100)

        for curve_fn, label, style in fitted_curves:
            plt.plot(dose_range, curve_fn(dose_range), label=label, **style)

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

def bernoulli_log_likelihood(logits, y, eps=1e-12):
    p = inverse_logit(logits)
    p = np.clip(p, eps, 1 - eps)

    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return ll

def log_likelihood_k_x0(params, dose, efficiency):
    k, x0 = params
    logit_eff = k * (dose - x0)

    ll = bernoulli_log_likelihood(logit_eff, efficiency)
    return ll

def log_likelihood_w_b(params, dose, efficiency):
    w, b = params
    logit_eff = w * dose + b
    
    ll = bernoulli_log_likelihood(logit_eff, efficiency)
    return ll

# McFadden R² calculation
def r2_score(ll_model, ll_null):
    r2 = 1 - (ll_model / ll_null)
    return r2

def p_value(ll_model, ll_null):
    lr_stat = 2 * (ll_model - ll_null)
    p_val = 1 - chi2.cdf(lr_stat, df=1)
    return p_val

def calculate_ll(params_1, params_2, dose, efficiency):
    param_null_k_x0 = (0, np.mean(dose))
    param_null_w_b = (0, logit(np.mean(efficiency.astype(float))))

    ll_model_k_x0 = log_likelihood_k_x0(params_1, dose, efficiency)
    ll_null_k_x0 = log_likelihood_k_x0(param_null_k_x0, dose, efficiency)

    ll_model_w_b = log_likelihood_w_b(params_2, dose, efficiency)
    ll_null_w_b = log_likelihood_w_b(param_null_w_b, dose, efficiency)
    return ll_model_k_x0, ll_null_k_x0, ll_model_w_b, ll_null_w_b

def gradient_descent_k_x0(params, dose, efficiency, learning_rate, num_iterations, tol=1e-12):
    k, x0 = params
    prev_cost = log_likelihood_k_x0((k, x0), dose, efficiency)
    for _ in range(num_iterations):
        logit_eff = k * (dose - x0)
        prob_eff = inverse_logit(logit_eff)

        error = efficiency - prob_eff

        dk = np.sum(error * (dose - x0))
        dx0 = -np.sum(error * k)

        k += learning_rate * dk
        x0 += learning_rate * dx0

        current_cost = log_likelihood_k_x0((k, x0), dose, efficiency)

        if abs(current_cost - prev_cost) < tol:
            print("\t\t\t\tConvergence in z = k*(x - x0), reached in {} iterations.".format(_))
            print("\t\t\t\tFinal parameters: k = {}, x0 = {}".format(k, x0))
            print("\t\t\t\tFinal Log-Likelihood: {}".format(current_cost))
            break

        prev_cost = current_cost

    return k, x0

def gradient_descent_w_b(params, dose, efficiency, learning_rate, num_iterations, tol=1e-12):
    w, b = params
    prev_cost = log_likelihood_w_b((w, b), dose, efficiency)
    for _ in range(num_iterations):
        logit_eff = w * dose + b
        prob_eff = inverse_logit(logit_eff)

        error = efficiency - prob_eff

        dw = np.sum(error * dose)
        db = np.sum(error)

        w += learning_rate * dw
        b += learning_rate * db

        current_cost = log_likelihood_w_b((w, b), dose, efficiency)

        if abs(current_cost - prev_cost) < tol:
            print("\t\t\t\tConvergence in z = w*x + b, reached in {} iterations.".format(_))
            print("\t\t\t\tFinal parameters: w = {}, b = {}".format(w, b))
            print("\t\t\t\tFinal Log-Likelihood: {}".format(current_cost))
            break

        prev_cost = current_cost

    return w, b

def run_regression(dose, efficiency, learning_rate_k_x0=0.01, learning_rate_w_b=0.3, num_iterations=10000):
    def transform_params_k_x0_to_w_b(params):
        k, x0 = params
        w = k
        b = -k * x0
        return (w, b)
    
    initial_params_k_x0 = (1, 0.1)  # Initial guess for k and x0
    initial_params_w_b = transform_params_k_x0_to_w_b(initial_params_k_x0)

    optimized_params_k_x0 = gradient_descent_k_x0(initial_params_k_x0, dose, efficiency, learning_rate_k_x0, num_iterations)
    optimized_params_w_b = gradient_descent_w_b(initial_params_w_b, dose, efficiency, learning_rate_w_b, num_iterations)
    ll_model_k_x0, ll_null_k_x0, ll_model_w_b, ll_null_w_b = calculate_ll(optimized_params_k_x0, optimized_params_w_b, dose, efficiency)

    r2_k_x0 = r2_score(ll_model_k_x0, ll_null_k_x0)
    p_val_k_x0 = p_value(ll_model_k_x0, ll_null_k_x0)
    r2_w_b = r2_score(ll_model_w_b, ll_null_w_b)
    p_val_w_b = p_value(ll_model_w_b, ll_null_w_b)

    optimized_params = [optimized_params_k_x0, optimized_params_w_b]

    print("\n----- Optimization Results -----")
    print("Model: sigmoid with shift parameterization (z = k*(x - x0))")
    print("Used learning rate:", learning_rate_k_x0)
    print(f"Optimized Parameters: k={optimized_params_k_x0[0]}, x0={optimized_params_k_x0[1]}")
    print(f"Final Log-Likelihood: {ll_model_k_x0}")
    print(f"McFadden R²: {r2_k_x0}")
    print(f"P-value: {p_val_k_x0}")
    print("--------------------------------")
    print("Model: sigmoid with linear parameterization (z = w*x + b)")
    print("Used learning rate:", learning_rate_w_b)
    print(f"Optimized Parameters: w={optimized_params_w_b[0]}, b={optimized_params_w_b[1]}")
    print(f"Final Log-Likelihood: {ll_model_w_b}")
    print(f"McFadden R²: {r2_w_b}")
    print(f"P-value: {p_val_w_b}")
    print("---------------------------------\n")

    if r2_w_b < r2_k_x0 and p_val_w_b > p_val_k_x0:
        print("Better model: restricted sigmoid function at z = k*(x - x0)")
    elif r2_k_x0 < r2_w_b and p_val_k_x0 > p_val_w_b:
        print("Better model: linear logit function at z = w*x + b")
    else:
        print("Models have mixed performance; further analysis may be needed.")

    return optimized_params

def main():
    optimal_params = run_regression(dose, efficiency, learning_rate_k_x0=0.01, learning_rate_w_b=0.3, num_iterations=20000)

    def fitted_curve_k_x0(x, params):
        return inverse_logit(params[0] * (x - params[1]))
    def fitted_curve_w_b(x, params):
        return inverse_logit(params[0] * x + params[1])

    curve_k_x0 = (partial(fitted_curve_k_x0, params=optimal_params[0]), "z = k*(x - x0)", {"color": "black", "linestyle": "dashed"})
    curve_w_b = (partial(fitted_curve_w_b, params=optimal_params[1]), "z = w*x + b", {"color": "green", "linestyle": "dotted"})
    
    plot_data(dose, efficiency, title='Fitted Curves', fitted_curves=[curve_k_x0, curve_w_b])


if __name__ == "__main__":
    main()




