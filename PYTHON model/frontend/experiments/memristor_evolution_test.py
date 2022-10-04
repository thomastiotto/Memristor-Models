import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

p100 = np.loadtxt('/Users/thomas/Desktop/+0.1V.csv', delimiter=',', usecols=1)

############ POWER-LAW MODEL ############

r_min = 2e2
r_max = 2.3e8
c = -0.146

iterations = 10

n = []
r = [p100[0]]

for i in range(iterations):
    n.append(((r[-1] - r_min) / r_max) ** (1 / c))
    r.append(r_min + r_max * (n[-1] + 1) ** c)
n.append(((r[-1] - r_min) / r_max) ** (1 / c))

fig, ax = plt.subplots()
ax.plot(p100, 'o', label='Data')
ax.plot(r, 'o', label='Power-law')

plt.title('Memristor resistance')
ax.set_xlabel(r"Pulse number $n$")
ax.set_ylabel(r"Resistance $R (\Omega)$")
ax.set_yscale("log")
ax.legend(loc='best')
ax.tick_params(axis='x')
ax.tick_params(axis='y')
plt.tight_layout()
fig.show()

print("Difference in resistances:")
print(np.abs(np.array(r) - np.array(r)))
print("Average difference in resistances:")
avg_err = np.sum(np.abs(np.array(r) - np.array(r))) / len(r)
print(avg_err, f"({(avg_err * 100) / (r_max - r_min)} %)")
print("MSE in resistances:")
print(mse(r, r))
