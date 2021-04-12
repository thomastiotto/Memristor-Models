import numpy as np
import scipy.signal


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
        self.print_equations()
        self.print_parameters()

    def print_equations(self, start="\t"):
        start_lv2 = start + "\t"
        print(start, "Equations:")
        print(start_lv2, "x(t) = w(t)/D")
        print(start_lv2, "V(t) = [ R_ON*x(t) + R_OFF*( 1-x(t) ) ]*I(t)*F(x)")
        print(start_lv2, "nu_D = dx/dt = ( mu_D*R_ON/D )*I(t)")

    def print_parameters(self, start="\t", simple=False):
        start_lv2 = start + "\t" if not simple else ""
        if not simple:
            print(start, "Parameters:")
            print(start_lv2, f"Device thickness D {self.D} m")
            print(start_lv2, f"Minimum resistance R_ON {self.R_ON} Ohm")
            print(start_lv2, f"Maximum resistance R_OFF {self.R_OFF} Ohm")
            print(start_lv2, f"Drift velocity of the oxygen deficiencies mu_D{self.m_D} m^2s^-1V^-1")
            print(start_lv2, f"Initial value of state variable x {self.x0} D")
        else:
            print([self.D, self.R_ON, self.R_OFF, self.m_D])


class InputVoltage():
    def __init__(self, shape, v_magnitude=1, period=1, t_max=0):
        input_functions = {
                "sine"    : self.sine,
                "triangle": self.triangle
                }

        assert shape in ["sine", "triangle"]
        if shape == "triangle": assert t_max > 0
        self.shape = shape
        self.func = input_functions[shape]
        self.voltage = v_magnitude
        self.t_max = t_max

    def sine(self, t):
        v = self.voltage * np.sin(2 * np.multiply(np.pi, t))

        return v

    def triangle(self, t):
        v = self.voltage * np.abs(scipy.signal.sawtooth(np.pi * t + np.pi / 2, 1 / 2))
        if isinstance(t, np.ndarray) and len(t) > 1:
            v[len(v) // 2:] *= -1
        elif t > self.t_max / 2:
            v *= -1

        return v

    def print(self, start="\t"):
        start_lv2 = start + "\t"
        print(f"{start_lv2}Shape {self.shape}")
        print(f"{start_lv2}Magnitude +/- {self.voltage} V")
        print(f"{start_lv2}Period 1 s")


class WindowFunction():
    def __init__(self, type, p=1, j=1):
        window_functions = {
                "none"    : self.no_window,
                "joglekar": self.joglekar,
                "biolek"  : self.biolek,
                "anusudha": self.anusudha,
                }

        assert type in ["none", "joglekar", "biolek", "anusudha"]
        self.type = type
        self.func = window_functions[type]
        self.p = p
        self.j = j

    def no_window(self, **kwargs):
        return 1

    def joglekar(self, **kwargs):
        x = kwargs["x"]

        return 1 - np.power(np.multiply(2, x) - 1, 2 * self.p)

    def biolek(self, **kwargs):
        x = kwargs["x"]
        i = kwargs["i"]

        return 1 - np.power(x - np.heaviside(-i, 1), 2 * self.p)

    def anusudha(self, **kwargs):
        x = kwargs["x"]

        return np.multiply(self.j, 1 - np.multiply(2, np.power(np.power(x, 3) - x + 1, self.p)))

    def print(self, start="\t"):
        start_lv2 = start + "\t"
        print(f"{start_lv2}Type {self.type}")
        print(f"{start_lv2}Parameter p {self.p}")
        if self.type in ("anusudha"):
            print(f"{start_lv2}Parameter j {self.j}")
