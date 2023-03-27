import numpy as np
import matplotlib.pyplot as plt

num_points = 100


def u(x, k):
    return np.sin(k * x)


def v(x, k):
    return np.sin(k * x)


def f_delta(x, k, l, delta):
    return Ku(x, k) + delta * np.sin(l * x)


def Ku(x, k):
    return sum(sigma_k(k) * inner_product(u(x, k), v(x, k), x) * u(x, k) for k in range(1, num_points))


def sigma_k(k):
    return np.exp(-k ** 2)


def inner_product(u, v, x):
    return (2 / np.pi) * np.trapz(u * v, x)


def k_a_L2(x, f_delta, v, u_k, sigma_k, alpha, k_max):
    return sum((1 / (sigma_k(k) + alpha * sigma_k(k) ** (-1))) * inner_product(f_delta, v(x, k), x) * u_k(x, k) for k in
               range(1, k_max + 1))


def k_a_dL2(x, f_delta, v, u_k, sigma_k, alpha, k_max):
    return sum(
        (1 / (sigma_k(k) + alpha * k ** 2 * sigma_k(k) ** (-1))) * inner_product(f_delta, v(x, k), x) * u_k(x, k) for k
        in range(1, k_max + 1))


k = 5
l = 7
delta = 1
num_points = 100
alpha_values = np.logspace(-2, 2, num_points)
x = np.linspace(0, 2 * np.pi, num_points)

bias_u = []
variance_u = []

bias_du = []
variance_du = []

deff_var = []
def_bias = []

clean_K_dagger_f =sum((sigma_k(k)) * inner_product(v(x, k), v(x, k), x) * u(x, k) for k in range(1, num_points+1))


for alpha in alpha_values:
    f_delta_current = f_delta(x, k, l, delta)

    u_alpha = k_a_L2(x, f_delta_current, v, u, sigma_k, alpha, k)
    bias = np.linalg.norm(u_alpha - u(x, k))
    variance = np.linalg.norm(u_alpha - Ku(x, k))

    bias_u.append(bias)
    variance_u.append(variance)

    u_alpha_d = k_a_dL2(x, f_delta_current, v, u, sigma_k, alpha, k)
    bias_d = np.linalg.norm(u_alpha_d - u(x, k))
    variance_d = np.linalg.norm(u_alpha_d - Ku(x, k))

    bias_du.append(bias_d)
    variance_du.append(variance_d)

    # deff_var.append(np.abs(bias_u-variance_d))
    # def_bias.append(np.abs(bias_u-bias_du))

plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.loglog(alpha_values, bias_u, label='Bias')
plt.loglog(alpha_values, variance_u, label='Variance')
plt.xlabel('Alpha')
plt.ylabel('Error')
plt.title('L2 u Regularization')
plt.legend()

plt.subplot(122)
plt.loglog(alpha_values, bias_du, label='Bias')
plt.loglog(alpha_values, variance_du, label='Variance')
plt.xlabel('Alpha')
plt.ylabel('Error')
plt.title('d_u Regularization')
plt.legend()

plt.show()
