import matplotlib.pyplot as plt
from block_timer.timer import Timer
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import curve_fit
from progressbar import progressbar

from window_functions import *


###############################################################################
#                                  ODE definition
###############################################################################

class hp_labs():
    def __init__(self, input, window_function, **kwargs):
        self.type = "HP Labs ion-drift model"

        self.input = input
        self.window_function = window_function
        self.V = input.func
        self.F = window_function.func

        self.D = kwargs["D"] if "D" in kwargs else 27e-9
        self.R_ON = kwargs["R_ON"] if "R_ON" in kwargs else 10e3
        self.R_OFF = kwargs["R_OFF"] if "R_OFF" in kwargs else 100e3
        self.m_D = kwargs["m_D"] if "m_D" in kwargs else 1e-14
        self.x0 = kwargs["x0"] if "x0" in kwargs else 0.1

    def I(self, t, x, *args):
        R_ON = args[0] if len(args) > 0 else self.R_ON
        R_OFF = args[1] if len(args) > 1 else self.R_OFF

        return self.V(t) / (np.multiply(R_ON, x) + np.multiply(R_OFF, (np.subtract(1, x))))

    def mu_D(self, t, x, *args):
        D = args[0] if len(args) > 0 else self.D
        R_ON = args[1] if len(args) > 1 else self.R_ON
        R_OFF = args[2] if len(args) > 2 else self.R_OFF
        m_D = args[3] if len(args) > 3 else self.m_D

        i = self.I(t, x, R_ON, R_OFF)

        return ((m_D * R_ON) / np.power(D, 2)) * i * self.F(x=x, i=i)

    def print(self):
        print(f"{self.type}:")
        print("\tEquations:")
        print(f"\t\tx(t) = w(t)/D")
        print("\t\tV(t) = [ R_ON*x(t) + R_OFF*( 1-x(t) ) ]*I(t)*F(x)")
        print("\t\tnu_D = dx/dt = ( mu_D*R_ON/D )*I(t)")
        print("\tInput V:")
        self.input.print()
        print("\tWindow F:")
        self.window_function.print()
        print("\tParameters:")
        print(f"\t\tDevice thickness {self.D} m")
        print(f"\t\tMinimum resistance {self.R_ON} Ohm")
        print(f"\t\tMaximum resistance {self.R_OFF} Ohm")
        print(f"\t\tDrift velocity of the oxygen deficiencies {self.m_D} m^2s^-1V^-1")
        print(f"\t\tInitial value of state variable x {self.x0} D")


###############################################################################
#                                  Setup
###############################################################################

## TIME
t_min = 0
t_max = 2
N = 10000
## INPUT
input_function_args = {
        "v_magnitude": 1,
        "t_max"      : t_max
        }
input_function = InputVoltage("sine", **input_function_args)
## WINDOW FUNCTION
window_function_args = {
        "p": 7,
        "j": 1
        }
window_function = WindowFunction("joglekar", **window_function_args)
## MEMRISTOR
memristor_args = {
        }

####

###############################################################################
#                         ODE simulation
###############################################################################

dt = (t_max - t_min) / N
time = np.arange(t_min, t_max + dt, dt)

memristor = hp_labs(input_function, window_function, **memristor_args)
memristor.print()
dxdt = memristor.mu_D
V = memristor.V
I = memristor.I
F = memristor.F
x0 = memristor.x0

x_euler = [x0]
x_rk4 = [x0]
current = [0.0]
solutions = []

print("Simulation:")
print(f"\tTime range [ {t_min},{t_max} ]")
print(f"\tSamples {N}")

# Solve ODE iteratively using Euler's method
with Timer(title="Euler"):
    print("Running Euler")

    for t in progressbar(time[:-1]):
        # print(
        #         "Euler", "t", '{:.6e}'.format( t ),
        #         "V", '{:.6e}'.format( V( t ) ),
        #         "I", '{:.6e}'.format( I( t, x_euler[ -1 ] ) ),
        #         "F", '{:.6e}'.format( F( x=x_euler[ -1 ], i=I( t, x_euler[ -1 ] ) ) ),
        #         "x", '{:.6e}'.format( x_euler[ -1 ] ),
        #         "dx", '{:.6e}'.format( dxdt( t, x_euler[ -1 ] ) )
        #         )

        current.append(I(t, x_euler[-1]))

        x_euler.append(x_euler[-1] + dxdt(t, x_euler[-1]) * dt)
    solutions.append((x_euler, time, "Euler"))

# Solve ODE iteratively using Runge-Kutta's method
with Timer(title="Runge-Kutta RK4"):
    print("Running Runge-Kutta RK4")
    for t in progressbar(time[:-1], redirect_stdout=True):
        current.append(I(t, x_rk4[-1]))

        k1 = dxdt(t, x_rk4[-1])
        k2 = dxdt(t + dt / 2, x_rk4[-1] + dt * k1 / 2)
        k3 = dxdt(t + dt / 2, x_rk4[-1] + dt * k2 / 2)
        k4 = dxdt(t + dt, x_rk4[-1] + dt * k3)
        x_rk4.append(x_rk4[-1] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    solutions.append((x_rk4, time, "Runge-Kutta"))

# Solve ODE with solver
with Timer(title="solve_ivp"):
    print("Running solve_ivp")
    x_solve_ivp = solve_ivp(dxdt, (t_min, t_max), [x0], method="LSODA", t_eval=time)
    solutions.append((x_solve_ivp.y[0, :], x_solve_ivp.t, "solve_ivp"))

# Plot simulated memristor
for x, t, title in solutions:
    v = V(t)
    i = I(t, x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax11 = axes[0]
    ax11.plot(t, i * 1e6, color="b")
    ax11.set_ylabel(r'Current ($\mu A$)', color='b')
    ax11.set_ylim([-30, 30])
    ax11.tick_params('y', colors='b')
    ax12 = ax11.twinx()
    ax12.plot(t, v, color="r")
    ax12.set_ylabel('Voltage (V)', color='r')
    ax12.tick_params('y', colors='r')
    ax12.set_ylim([-1.5, 1.5])
    ax2 = axes[1]
    ax2.plot(v, i * 1e6)
    ax2.set_ylim([-25, 25])
    ax2.set_ylabel(r'Current ($\mu A$)')
    ax2.set_xlabel('Voltage (V)')
    fig.suptitle(f"Memristor Voltage and Current vs. Time ({title})")
    fig.tight_layout()
    fig.show()

####

###############################################################################
#                       Data sampling to solve_ivp solution
###############################################################################

percentage_samples = 100
noise_percentage = 10

np.random.seed(1729)

simulated_data = I(x_solve_ivp.t, x_solve_ivp.y[0, :])
num_samples = int(percentage_samples * len(simulated_data) / 100)
noisy_solution = np.random.normal(simulated_data, np.abs(simulated_data) * noise_percentage / 100,
                                  size=simulated_data.size)
# TODO I need to generate k data points from model to fit, as each sample need to correspond to a given time
# TODO alternatively, create a list of tuples (t,data)
samples = np.random.choice(np.squeeze(noisy_solution), num_samples, replace=False)


####

###############################################################################
#                         ODE fitting
###############################################################################

def ode_fitting(t, D, R_ON, R_OFF, m_D):
    # call solve_ivp() on dxdt with R_INIT and parameters

    sol = solve_ivp(dxdt, (t_min, t_max), [x0], method="LSODA",
                    t_eval=t,
                    args=(D, R_ON, R_OFF, m_D))

    return I(t, sol.y[0, :])

# popt, pcov = curve_fit( ode_fitting, time, noisy_solution )
# print( popt )

####

###############################################################################
#                         Residuals
###############################################################################


####
