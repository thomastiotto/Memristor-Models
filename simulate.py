import matplotlib.pyplot as plt
from block_timer.timer import Timer
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from progressbar import progressbar
import scipy.stats as stats
from order_of_magnitude import order_of_magnitude
import os
import multiprocessing as mp

from functions import *
from models import *
from experiments import *

###############################################################################
#                                  Setup
###############################################################################

experiment = oblea_pulsed()

time = experiment.simulation["time"]
dt = experiment.simulation["dt"]
x0 = experiment.simulation["x0"]
dxdt = experiment.functions["dxdt"]
V = experiment.functions["V"]
I = experiment.functions["I"]

####

###############################################################################
#                         ODE simulation
###############################################################################
solver = "LSODA"

solutions = []

# # Solve ODE iteratively using Euler's method
# x_euler = euler_solver(dxdt, time, dt, x0)
# solutions.append((x_euler, time, "Euler"))

# # Solve ODE iteratively using Runge-Kutta's method
# x_rk4 = rk4_solver(dxdt, time, dt, x0)
# solutions.append((x_rk4, time, "Runge-Kutta"))

# Solve ODE with solver
with Timer(title="solve_ivp"):
    print("Running solve_ivp")
    x_solve_ivp = solve_ivp(dxdt, (time[0], time[-1]), [x0], method=solver, t_eval=time)
    solutions.append((x_solve_ivp.y[0, :], x_solve_ivp.t, "solve_ivp"))

# Plot simulated memristor behaviour
for x, t, title in solutions:
    v = V(t)
    i = I(t, x)

    fig = plot_memristor(v, i, t, "simulated")
    fig.show()

    # make video of simulation
    if not os.path.exists(f"{experiment.name}.mp4"):
        if __name__ == "__main__":
            mp.set_start_method("fork")
            p = mp.Process(target=plot_memristor,
                           args=(v, i, t, title, solver, True, experiment.name))
            p.start()
####


###############################################################################
#                       Data sampling to solve_ivp solution
###############################################################################

np.random.seed(1729)
noise_percentage = 10

# Generate noisy data from memristor model
simulated_data = I(x_solve_ivp.t, x_solve_ivp.y[0, :])
noisy_solution = np.random.normal(simulated_data, np.abs(simulated_data) * noise_percentage / 100,
                                  size=simulated_data.size)

# Plot noisy data
fig2 = plot_memristor(V(x_solve_ivp.t), noisy_solution, x_solve_ivp.t, "noisy")
fig2.show()


####

###############################################################################
#                         ODE fitting
###############################################################################

def ode_fitting(t, a1, a2, b, Ap, An, Vp, Vn, alphap, alphan, xp, xn, eta):
    # call solve_ivp() on dxdt with R_INIT and parameters

    sol = solve_ivp(dxdt, (t[0], t[-1]), [x0], method="LSODA",
                    t_eval=t,
                    args=[a1, a2, b, Ap, An, Vp, Vn, alphap, alphan, xp, xn, eta]
                    # p0=[0]
                    )

    # print("******************")
    # print("D", D, "R_ON", R_ON, "R_OFF", R_OFF, "m_D", m_D)
    # print("Mean x", np.mean(sol.y[0, :]))
    # print("Mean current I(t,x)", np.mean(I(t, sol.y[0, :])))

    return I(t, sol.y[0, :])


# Fit parameters to noisy data
with Timer(title="curve_fit"):
    print("Running curve_fit")
    popt, pcov = curve_fit(ode_fitting, time, noisy_solution,
                           # bounds=(0, [1e-7, 1e4, 1e5, 1e-13]),
                           # p0=[10e-9, 10e3, 100e3, 1e-14]
                           )

# with Timer(title="lmfit"):
#     print("Running lmfit")
#     popt, pcov = curve_fit(ode_fitting, time, noisy_solution,
#                            bounds=(0, [1e-7, 1e4, 1e5, 1e-13])
#                            )


print("curve_fit parameters", end=" ")
experiment.memristor.print_parameters(start="", simple=True)
print("Fitted parameters", popt)

# Simulate memristor with fitted parameters
with Timer(title="solve_ivp"):
    print("Running solve_ivp")
    x_solve_ivp_fitted = solve_ivp(dxdt, (time[0], time[-1]), [x0], method="LSODA", t_eval=time,
                                   args=popt
                                   )

# Plot reconstructed data
fig3, axes = plt.subplots(1, 2, figsize=(10, 4))
t = x_solve_ivp_fitted.t
v = V(t)
fitted_data = I(t, x_solve_ivp_fitted.y[0, :])
ax11 = axes[0]
ax11.plot(t, fitted_data, color="b")
ax11.set_ylabel(r'Current ($\mu A$)', color='b')
# ax11.set_ylim([-30, 30])
ax11.tick_params('y', colors='b')
ax12 = ax11.twinx()
ax12.plot(t, v, color="r")
ax11.set_xlabel('Time (s)')
ax12.set_ylabel('Voltage (V)', color='r')
ax12.tick_params('y', colors='r')
# ax12.set_ylim([-1.5, 1.5])
ax2 = axes[1]
ax2.plot(v, fitted_data)
# ax2.set_ylim([-25, 25])
ax2.set_xlabel('Voltage (V)')
ax2.set_ylabel(r'Current ($\mu A$)')
fig3.suptitle(f"Reconstructed current vs. Time")
fig3.tight_layout()
# fig3.show()

####


###############################################################################
#                         Error
###############################################################################
error = np.sum(np.abs(simulated_data[1:] - fitted_data[1:]))
error_average = np.mean(error)
error_percent = error / fitted_data[1:] * 100
print(f"Average error {order_of_magnitude.symbol(error_average)}A ({np.mean(error_percent):.2f}%)")

###############################################################################
#                         Residuals
###############################################################################

residuals = noisy_solution - fitted_data
fig4, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(fitted_data, residuals)
axes[0].set_xlabel("Residuals")
axes[0].set_ylabel("Fitted values")
axes[0].set_title("Residuals")
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residuals")
fig4.suptitle(f"Residual analysis")
fig4.tight_layout()
# fig4.show()
####
