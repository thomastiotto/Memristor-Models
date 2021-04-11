import numpy as np
import scipy.signal


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
        print(f"{start_lv2}Magnitude +/-{self.voltage} V")
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
