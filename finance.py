import numpy as np
from matplotlib import pyplot as plt


def geometric_brownian_motion_dS(S, sigma):
    return sigma * S * np.random.normal()


time = np.linspace(start=0, stop=1, num=10000)

S = np.zeros(len(time))
S[0] = 100
for i in range(1, len(time)):
    S[i] = S[i - 1] + geometric_brownian_motion_dS(S[i - 1], 0.003)

strike = 90 * np.ones((len(time)))

# filter out all S above strike price
S_above = np.where(S > strike, S, np.nan)
plt.plot(time, S_above, "g")
S_below = np.where(S <= strike, S, np.nan)
plt.plot(time, S_below, "r")

plt.xlabel("Time")
plt.ylabel("Price")

plt.plot(time, strike,"black", label="Strike Price")

plt.legend()
plt.show()


time = np.linspace(start=0, stop=1, num=1000)
for k in range(10):
    S = np.zeros((10, len(time),))
    S[k][0] = 100
    for i in range(1, len(time)):
        S[k][i] = S[i - 1] + geometric_brownian_motion_dS(S[k][i - 1], 0.003)
        plt.plot(time, S[k])
plt.show()
