import matplotlib.pyplot as plt
from block_timer.timer import Timer
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from progressbar import progressbar
import scipy.stats as stats
from order_of_magnitude import order_of_magnitude

from functions import *
from models import *

###############################################################################
#                                  Setup
###############################################################################

## TIME
t_max = 50e-3
frequency = 100e3

## INPUT
input_function_args = {
        "shape"    : "triangle",
        "frequency": 100,
        "vp"       : 0.45,
        "t_max"    : t_max,
        }
input_function = InputVoltage(**input_function_args)

## MEMRISTOR
memristor_args = {
        }

## OTHER
solver = "RK23"

####

###############################################################################
#                         ODE simulation
###############################################################################
t_min = 0
dt = 1 / frequency
N = (t_max - t_min) * frequency
time = np.arange(t_min, t_max + dt, dt)

memristor = Yakopcic(input_function, **memristor_args)
memristor.print()
dxdt = memristor.dxdt
V = memristor.V
I = memristor.I
x0 = memristor.x0

x_euler = [x0]
x_rk4 = [x0]
current = [0.0]
solutions = []

print("Simulation:")
print(f"\tTime range [ {t_min}, {t_max} ]")
print(f"\tSamples {N}")

# Solve ODE iteratively using Euler's method
# with Timer(title="Euler"):
#     print("Running Euler")
#
#     for t in progressbar(time[:-1]):
#         # print(
#         #         "Euler", "t", '{:.6e}'.format( t ),
#         #         "V", '{:.6e}'.format( V( t ) ),
#         #         "I", '{:.6e}'.format( I( t, x_euler[ -1 ] ) ),
#         #         "F", '{:.6e}'.format( F( x=x_euler[ -1 ], i=I( t, x_euler[ -1 ] ) ) ),
#         #         "x", '{:.6e}'.format( x_euler[ -1 ] ),
#         #         "dx", '{:.6e}'.format( dxdt( t, x_euler[ -1 ] ) )
#         #         )
#
#         current.append(I(t, x_euler[-1]))
#
#         x_euler.append(x_euler[-1] + dxdt(t, x_euler[-1]) * dt)
#     solutions.append((x_euler, time, "Euler"))

# Solve ODE iteratively using Runge-Kutta's method
# with Timer(title="Runge-Kutta RK4"):
#     print("Running Runge-Kutta RK4")
#     for t in progressbar(time[:-1], redirect_stdout=True):
#         current.append(I(t, x_rk4[-1]))
#
#         k1 = dxdt(t, x_rk4[-1])
#         k2 = dxdt(t + dt / 2, x_rk4[-1] + dt * k1 / 2)
#         k3 = dxdt(t + dt / 2, x_rk4[-1] + dt * k2 / 2)
#         k4 = dxdt(t + dt, x_rk4[-1] + dt * k3)
#         x_rk4.append(x_rk4[-1] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
#     solutions.append((x_rk4, time, "Runge-Kutta"))

# Solve ODE with solver
with Timer(title="solve_ivp"):
    print("Running solve_ivp")
    x_solve_ivp = solve_ivp(dxdt, (time[0], time[-1]), [x0], method=solver, t_eval=time)
    solutions.append((x_solve_ivp.y[0, :], x_solve_ivp.t, "solve_ivp"))

# Plot simulated memristor behaviour
for x, t, title in solutions:
    v = V(t)
    i = I(t, x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax11 = axes[0]
    max_oom = order_of_magnitude.power_of_ten(np.max(i))
    ax11.plot(t, i * max_oom, color="b")
    # # Plot voltage threshold points
    # for j in range(len(v)):
    #     try:
    #         if v[j] <= memristor.Vp <= v[j + 1]:
    #             print(f"[t={t[j + 1]}] Positive Vp={memristor.Vp} exceeded, x {x[j]} -> {x[j + 1]}")
    #             ax11.plot(t[j + 1], i[j + 1] * max_oom, "ro")
    #         if v[j] >= -memristor.Vn >= v[j + 1]:
    #             print(f"[t={t[j + 1]}] Negative Vn={memristor.Vn} exceeded, x {x[j]} -> {x[j + 1]}")
    #             ax11.plot(t[j + 1], i[j + 1] * max_oom, "bo")
    #     except:
    #         pass
    ax11.set_ylabel(f'Current ({order_of_magnitude.symbol(max_oom, omit_x=True)}A)', color='b')
    ax11.tick_params('y', colors='b')
    ax12 = ax11.twinx()
    ax12.plot(t, v, color="r")
    # ax12.axhline(memristor.Vp, color="r")
    # ax12.annotate(f"Vp={memristor.Vp}", (t[-1], memristor.Vp), color="r")
    # ax12.axhline(-memristor.Vn, color="r")
    # ax12.annotate(f"Vn={memristor.Vn}", (t[-1], -memristor.Vn), color="r")
    ax11.set_xlabel('Time (s)')
    ax12.set_ylabel('Voltage (V)', color='r')
    ax12.tick_params('y', colors='r')
    ax12.set_ylim([np.min(v) - np.abs(0.5 * np.min(v)), np.max(v) + np.abs(0.5 * np.max(v))])
    ax2 = axes[1]
    ax2.plot(v, i * max_oom)
    arrows_every = len(v) // 200 if len(v) > 200 else 1
    x1 = np.ma.masked_array(v[:-1:arrows_every], (np.diff(v) > 0)[::arrows_every])
    x2 = np.ma.masked_array(v[:-1:arrows_every], (np.diff(v) < 0)[::arrows_every])
    ax2.plot(x1, (i * max_oom)[:-1:arrows_every], 'b<')
    ax2.plot(x2, (i * max_oom)[:-1:arrows_every], 'r>')
    # ax2.axvline(memristor.Vp, color="r")
    # ax2.annotate(f"Vp={memristor.Vp}", (memristor.Vp, i[-1]), color="r")
    # ax2.axvline(-memristor.Vn, color="b")
    # ax2.annotate(f"Vn={memristor.Vn}", (-memristor.Vn, i[-1]), color="b")
    ax2.set_ylabel(f'Current ({order_of_magnitude.symbol(max_oom, omit_x=True)}A)')
    ax2.set_xlabel('Voltage (V)')
    fig.suptitle(f"Memristor Voltage and Current vs. Time ({title}" + f" - {solver})" if title == "solve_ivp" else ")")
    fig.tight_layout()
    fig.show()

    # # Plot simulated memristor behaviour
    # for x, t, title in solutions:
    #     i = I(t, x)
    #
    #     fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    #     ax11 = axes
    #     max_oom = order_of_magnitude.power_of_ten(np.max(i))
    #     ax11.plot(t, i * max_oom, color="b")
    #     ax11.set_ylabel(f'Current ({order_of_magnitude.symbol(max_oom, omit_x=True)}A)', color='b')
    #     ax11.tick_params('y', colors='b')
    #     ax12 = ax11.twinx()
    #     ax12.plot(t, x, color="g")
    #     ax12.axhline(memristor.xp, color="g")
    #     ax12.annotate(f"xp={memristor.xp}", (t[-1], memristor.xp), color="g")
    #     ax12.axhline(1 - memristor.xn, color="g")
    #     ax12.annotate(f"xn={1 - memristor.xn}", (t[-1], 1 - memristor.xn), color="g")
    #     ax11.set_xlabel('Time (s)')
    #     ax12.set_ylabel('State x', color="g")
    #     ax12.tick_params('y', colors="g")
    #     ax12.set_ylim([np.min(x) - np.abs(0.5 * np.min(x)), np.max(x) + np.abs(0.5 * np.max(x))])
    #     fig.suptitle(
    #             f"Memristor Current and State Variable vs. Time ({title}" + f" - {solver})" if title == "solve_ivp"
    #             else ")")
    #     fig.tight_layout()
    #     fig.show()

    # # Plot simulated memristor behaviour
    # for x, t, title in solutions:
    #     v = V(t)
    #
    #     fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    #     ax11 = axes
    #     max_oom = order_of_magnitude.power_of_ten(np.max(i))
    #     ax11.plot(t, v, color="r")
    #     ax11.axhline(memristor.Vp, color="r")
    #     ax11.annotate(f"Vp={memristor.Vp}", (t[-1], memristor.Vp), color="r")
    #     ax11.axhline(-memristor.Vn, color="r")
    #     ax11.annotate(f"Vn={memristor.Vn}", (t[-1], -memristor.Vn), color="r")
    #     ax11.axhline(0, color="r")
    #     ax11.set_ylabel("Voltage (V)", color="r")
    #     ax11.tick_params("y", colors="r")
    #     ax12 = ax11.twinx()
    #     ax12.plot(t, x, color="g")
    #     ax12.axhline(memristor.xp, color="g")
    #     ax12.annotate(f"xp={memristor.xp}", (t[-1], memristor.xp), color="g")
    #     ax12.axhline(1 - memristor.xn, color="g")
    #     ax12.annotate(f"xn={1 - memristor.xn}", (t[-1], 1 - memristor.xn), color="g")
    #     ax11.set_xlabel('Time (s)')
    #     ax12.set_ylabel('State x', color="g")
    #     ax12.tick_params("y", colors="g")
    #     ax12.set_ylim([np.min(x) - np.abs(0.5 * np.min(x)), np.max(x) + np.abs(0.5 * np.max(x))])
    #     fig.suptitle(
    #             f"Memristor Voltage and State Variable vs. Time ({title}" + f" - {solver})" if title == "solve_ivp"
    #             else ")")
    #     fig.tight_layout()
    #     fig.show()

####


###############################################################################
#                       Data sampling to solve_ivp solution
###############################################################################

percentage_samples = 100
noise_percentage = 10

np.random.seed(1729)

# Generate noisy data from memristor model
simulated_data = I(x_solve_ivp.t, x_solve_ivp.y[0, :])
num_samples = int(percentage_samples * len(simulated_data) / 100)
noisy_solution = np.random.normal(simulated_data, np.abs(simulated_data) * noise_percentage / 100,
                                  size=simulated_data.size)
# TODO I need to generate k data points from model to fit, as each sample need to correspond to a given time
# TODO alternatively, create a list of tuples (t,data)
samples = np.random.choice(np.squeeze(noisy_solution), num_samples, replace=False)

# Plot noisy data
fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
ax11 = axes[0]
ax11.plot(x_solve_ivp.t, noisy_solution * 1e6, color="b")
ax11.set_ylabel(r'Current ($\mu A$)', color='b')
# ax11.set_ylim([-30, 30])
ax11.tick_params('y', colors='b')
ax12 = ax11.twinx()
ax12.plot(x_solve_ivp.t, V(x_solve_ivp.t), color="r")
ax11.set_xlabel('Time (s)')
ax12.set_ylabel('Voltage (V)', color='r')
ax12.tick_params('y', colors='r')
# ax12.set_ylim([-1.5, 1.5])
ax2 = axes[1]
ax2.plot(V(x_solve_ivp.t), noisy_solution * 1e6)
# ax2.set_ylim([-25, 25])
ax2.set_ylabel(r'Current ($\mu A$)')
ax2.set_xlabel('Voltage (V)')
fig2.suptitle(f"Noisy current vs. Time")
fig2.tight_layout()


# fig2.show()


####

###############################################################################
#                         ODE fitting
###############################################################################

def ode_fitting(t, D, R_ON, R_OFF, m_D):
    # call solve_ivp() on dxdt with R_INIT and parameters

    sol = solve_ivp(dxdt, (t[0], t[-1]), [x0], method="LSODA",
                    t_eval=t,
                    args=[D, R_ON, R_OFF, m_D],
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
memristor.print_parameters(start="", simple=True)
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
error = np.abs(simulated_data[1:] - fitted_data[1:])
error_percent = error / fitted_data[1:] * 100
print(f"Average error {order_of_magnitude.oom(np.mean(error))}A ({np.mean(error_percent):.2f}%)")

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
