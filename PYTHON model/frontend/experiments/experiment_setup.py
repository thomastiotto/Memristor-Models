import numpy as np
import functions
import yakopcic_model


class Experiment:
    def __init__(self, sim_args, model, memristor_args):
        self.name = None
        self.dt = sim_args["dt"]

        self.simulation = {"x0": sim_args["x0"], "dt": self.dt}
        self.memristor_args = memristor_args
        self.memristor_args.update({"x0": sim_args["x0"]})
        self.memristor = model(**self.memristor_args)
        # self.memristor.print()

        self.functions = {
            "dxdt": self.memristor.dxdt,
            "I": self.memristor.I,
        }

        # print("Simulation:")
        # print(f"\tTime range [ {self.t_min}, {self.t_max} ]")
        # print(f"\tSamples {self.simulation['N']}")
        # print(f"\tInitial value of state variable {self.simulation['x0']}")


class YakopcicSET(Experiment):
    def __init__(self):
        super(YakopcicSET, self).__init__(
            sim_args={"dt": 0.001, "x0": 0.0},
            model=yakopcic_model.YakopcicNew,
            memristor_args={
                "Ap": 9,
                "An": 1,
                "Vp": 0.5,
                "Vn": 0.5,
                "alphap": 1,
                "alphan": 1,
                "xp": 0.1,
                "xn": 0.242,
                "eta": 1
            },
        )
        self.name = "SET"
