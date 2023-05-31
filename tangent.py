import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return np.sin(x) + x * 0.4


def df_dx(x):
    return np.cos(x) + 0.4


def linear(x, m, t):
    return m * x + t

fig, axs = plt.subplots(nrows=1, ncols=2)

x = np.linspace(start=0, stop=2 * np.pi)
y = f(x)

s = np.array([1, 3, 5])
y_s = f(s)

axs[0].plot(x, y, label="Underlying Function", color="black", linestyle='dashed')
axs[0].scatter(s, y_s, color="r", marker="x", label="Training Data")

coefficients = np.polyfit(s, y_s, 3)
y_pred = np.polyval(coefficients, x)
axs[0].plot(x, y_pred, color="b", label="Fit")

axs[1].plot(x, y, label="Underlying Function", color="black", linestyle='dashed')
axs[1].scatter(s, y_s, color="r", marker="x", label="Training Data")

s_diff = []
y_s_diff = []
dx = 0.1

plot_offset = 0.5
for s_i, y_s_i in zip(s, y_s):
    d_y_s_i = df_dx(s_i)
    plot_range = np.linspace(start=s_i - plot_offset, stop= s_i + plot_offset)
    plot_linear = linear(plot_range, m=d_y_s_i, t=y_s_i-d_y_s_i*s_i)
    plt.plot(plot_range, plot_linear, color="r")

    s_diff.append(s_i + dx)
    y_s_diff.append(y_s_i + d_y_s_i * dx)
    s_diff.append(s_i - dx)
    y_s_diff.append(y_s_i - d_y_s_i * dx)

s_and_diff = np.append(s, s_diff)
y_s_and_diff = np.append(y_s, y_s_diff)

coefficients = np.polyfit(s_and_diff, y_s_and_diff, 3)
y_pred = np.polyval(coefficients, x)
axs[1].plot(x, y_pred, color="b", label="Fit")

# set label
axs[0].set(xlabel='x', ylabel='y', title="Fit without differentials")
axs[1].set(xlabel='x', ylabel='y', title="Fit with differentials")
axs[1].legend()
fig.savefig('tangents.png')
fig.tight_layout(w_pad=1)
fig.show()

