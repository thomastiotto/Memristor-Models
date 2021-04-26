import numpy as np
import functions
import models


class hp_labs_sine():

    def __init__(self):
        self.name = "HP Labs sine"

        ## TIME
        t_max = 2  # Simulation time
        frequency = 100e3
        t_min = 0
        dt = 1 / frequency
        N = (t_max - t_min) * frequency
        self.simulation = {
                "t_min": 0,
                "dt"   : dt,
                "N"    : N,
                "time" : np.arange(t_min, t_max + dt, dt),
                "x0"   : 0.1
                }

        ## MEMRISTOR
        input_function_args = {
                "shape"    : "sine",
                "frequency": 1,
                "vp"       : 1,
                "t_max"    : t_max,
                }
        input_function = functions.InputVoltage(**input_function_args)

        window_function_args = {
                "p": 7,
                "j": 1
                }
        window_function = functions.WindowFunction("joglekar", **window_function_args)

        memristor_args = {
                "RON" : 10e3,
                "ROFF": 100e3,
                "D"   : 27e-9,
                "muD" : 1e-14
                }

        self.memristor = models.HPLabs(input_function, window_function, **memristor_args)
        self.memristor.print()

        self.functions = {
                "dxdt": self.memristor.mu_D,
                "V"   : self.memristor.V,
                "I"   : self.memristor.I,
                }

        print("Simulation:")
        print(f"\tTime range [ {t_min}, {t_max} ]")
        print(f"\tSamples {N}")
        print(f"\tInitial value of state variable {self.simulation['x0']}")


class hp_labs_pulsed():

    def __init__(self):
        self.name = "HP Labs pulsed"

        ## TIME
        t_max = 8  # Simulation time
        frequency = 100e3
        t_min = 0
        dt = 1 / frequency
        N = (t_max - t_min) * frequency
        self.simulation = {
                "t_min": 0,
                "dt"   : dt,
                "N"    : N,
                "time" : np.arange(t_min, t_max + dt, dt),
                "x0"   : 0.093,
                }

        ## MEMRISTOR
        input_function_args = {
                "shape"    : "triangle",
                "frequency": 0.5,
                "vp"       : 1,
                "t_max"    : t_max,
                }
        input_function = functions.InputVoltage(**input_function_args)

        window_function_args = {
                "p": 2,
                "j": 1
                }
        window_function = functions.WindowFunction("joglekar", **window_function_args)

        memristor_args = {
                "RON" : 1e3,
                "ROFF": 10e3,
                "D"   : 85e-9,
                "muD" : 2e-14
                }

        self.memristor = models.HPLabs(input_function, window_function, **memristor_args)
        self.memristor.print()

        self.functions = {
                "dxdt": self.memristor.mu_D,
                "V"   : self.memristor.V,
                "I"   : self.memristor.I,
                }

        print("Simulation:")
        print(f"\tTime range [ {t_min}, {t_max} ]")
        print(f"\tSamples {N}")
        print(f"\tInitial value of state variable {self.simulation['x0']}")


class oblea_sine():

    def __init__(self):
        self.name = "Oblea sine"

        ## TIME
        t_max = 40e-3
        frequency = 100e3
        t_min = 0
        dt = 1 / frequency
        N = (t_max - t_min) * frequency
        self.simulation = {
                "t_min": 0,
                "dt"   : dt,
                "N"    : N,
                "time" : np.arange(t_min, t_max + dt, dt),
                "x0"   : 0.11
                }

        ## INPUT
        input_function_args = {
                "shape"    : "sine",
                "frequency": 100,
                "vp"       : 0.45,
                "t_max"    : t_max,
                }
        input_function = functions.InputVoltage(**input_function_args)

        ## MEMRISTOR
        memristor_args = {
                "a1"    : 0.17,
                "a2"    : 0.17,
                "b"     : 0.05,
                "Ap"    : 4000,
                "An"    : 4000,
                "Vp"    : 0.16,
                "Vn"    : 0.15,
                "alphap": 1,
                "alphan": 5,
                "xp"    : 0.3,
                "xn"    : 0.5,
                "eta"   : 1,
                }

        # SETUP

        self.memristor = models.Yakopcic(input_function, **memristor_args)
        self.memristor.print()

        self.functions = {
                "dxdt": self.memristor.dxdt,
                "V"   : self.memristor.V,
                "I"   : self.memristor.I,
                }

        print("Simulation:")
        print(f"\tTime range [ {t_min}, {t_max} ]")
        print(f"\tSamples {N}")
        print(f"\tInitial value of state variable {self.simulation['x0']}")


class oblea_pulsed():

    def __init__(self):
        self.name = "Oblea pulsed"

        ## TIME
        t_max = 50e-3
        frequency = 100e3
        t_min = 0
        dt = 1 / frequency
        N = (t_max - t_min) * frequency
        self.simulation = {
                "t_min": 0,
                "dt"   : dt,
                "N"    : N,
                "time" : np.arange(t_min, t_max + dt, dt),
                }

        ## INPUT
        input_function_args = {
                "shape"    : "triangle",
                "frequency": 100,
                "vp"       : 0.25,
                "t_max"    : t_max,
                }
        input_function = functions.InputVoltage(**input_function_args)

        ## MEMRISTOR
        memristor_args = {
                "a1"    : 0.097,
                "a2"    : 0.097,
                "b"     : 0.05,
                "x0"    : 0.001,
                "Ap"    : 4000,
                "An"    : 4000,
                "Vp"    : 0.16,
                "Vn"    : 0.15,
                "alphap": 1,
                "alphan": 5,
                "xp"    : 0.3,
                "xn"    : 0.5,
                "eta"   : 1,
                }

        # SETUP

        self.memristor = models.Yakopcic(input_function, **memristor_args)
        self.memristor.print()

        self.functions = {
                "dxdt": self.memristor.dxdt,
                "V"   : self.memristor.V,
                "I"   : self.memristor.I,
                "x0"  : self.memristor.x0
                }

        print("Simulation:")
        print(f"\tTime range [ {t_min}, {t_max} ]")
        print(f"\tSamples {N}")
