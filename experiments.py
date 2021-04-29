import numpy as np
from scipy.integrate import solve_ivp

import functions
import models


class Experiment():
    def __init__( self, sim_args, model, memristor_args, input_args, window_function_args=None ):
        self.name = None
        t_max = sim_args[ "t_max" ]
        frequency = sim_args[ "frequency" ]
        
        t_min = 0
        dt = 1 / frequency
        N = (t_max - t_min) * frequency
        
        self.simulation = {
                "t_min": 0,
                "dt"   : dt,
                "N"    : N,
                "time" : np.arange( t_min, t_max + dt, dt ),
                "x0"   : sim_args[ "x0" ]
                }
        
        self.input_args = input_args
        self.input_args.update( { "t_max": t_max, "vn": input_args[ "vp" ] } )
        self.window_function_args = window_function_args
        
        self.input_function = functions.InputVoltage( **self.input_args )
        self.window_function = functions.WindowFunction(
                **self.window_function_args ) if self.window_function_args else None
        
        self.memristor_args = memristor_args
        self.memristor_args.update( { "x0": sim_args[ "x0" ] } )
        
        self.memristor = model( self.input_function, **self.memristor_args ) \
            if not self.window_function \
            else model( self.input_function, self.window_function, **self.memristor_args )
        
        # important for fitting as we can't pass kwargs
        assert self.memristor.parameters() == list( self.memristor.passed_parameters.keys() )
        
        self.memristor.print()
        
        self.functions = {
                "dxdt": self.memristor.dxdt,
                "V"   : self.memristor.V,
                "I"   : self.memristor.I,
                }
        
        self.fitting = {
                "noise": 10
                }
        
        print( "Simulation:" )
        print( f"\tTime range [ {t_min}, {t_max} ]" )
        print( f"\tSamples {N}" )
        print( f"\tInitial value of state variable {self.simulation[ 'x0' ]}" )
    
    def fit_memristor( self ):
        pass


class hp_labs_sine( Experiment ):
    
    def __init__( self ):
        super( hp_labs_sine, self ).__init__(
                sim_args={ "t_max": 2, "frequency": 100e3, "x0": 0.1 },
                model=models.HPLabs,
                memristor_args={ "D": 27e-9, "RON": 10e3, "ROFF": 100e3, "muD": 1e-14 },
                input_args={ "shape": "sine", "frequency": 1, "vp": 1 },
                window_function_args={ "type": "joglekar", "p": 7, "j": 1 }
                )
        
        self.name = "HP Labs sine"
        self.fitting.update( { "bounds": (0, [ 1e-7, 1e4, 1e5, 1e-13 ]) } )


class hp_labs_pulsed( Experiment ):
    
    def __init__( self ):
        super( hp_labs_pulsed, self ).__init__(
                sim_args={ "t_max": 8, "frequency": 100e3, "x0": 0.093 },
                model=models.HPLabs,
                memristor_args={ "D": 85e-9, "RON": 1e3, "ROFF": 10e3, "muD": 2e-14 },
                input_args={ "shape": "triangle", "frequency": 0.5, "vp": 1 },
                window_function_args={ "type": "joglekar", "p": 2, "j": 1 }
                )
        
        self.name = "HP Labs pulsed"
        self.fitting.update( { "bounds": (0, [ 1e-7, 1e4, 1e5, 1e-13 ]) } )


class oblea_sine( Experiment ):
    
    def __init__( self ):
        super( oblea_sine, self ).__init__(
                sim_args={ "t_max": 40e-3, "frequency": 100e3, "x0": 0.11 },
                model=models.Yakopcic,
                memristor_args={ "a1"    : 0.17,
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
                                 "eta"   : 1
                                 },
                input_args={ "shape": "sine", "frequency": 100, "vp": 0.45 },
                )
        
        self.name = "Oblea sine"
        self.fitting.update(
                { "bounds": ([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 ], [ 1, 1, 1, 1e4, 1e4, 1, 1, 1, 1, 1, 1, 1 ]) } )


class oblea_pulsed( Experiment ):
    def __init__( self ):
        super( oblea_pulsed, self ).__init__(
                sim_args={ "t_max": 50e-3, "frequency": 100e3, "x0": 0.001 },
                model=models.Yakopcic,
                memristor_args={
                        "a1"    : 0.097,
                        "a2"    : 0.097,
                        "b"     : 0.05,
                        "Ap"    : 4000,
                        "An"    : 4000,
                        "Vp"    : 0.16,
                        "Vn"    : 0.15,
                        "alphap": 1,
                        "alphan": 5,
                        "xp"    : 0.3,
                        "xn"    : 0.5,
                        "eta"   : 1
                        },
                input_args={ "shape": "triangle", "frequency": 100, "vp": 0.25 },
                )
        
        self.name = "Oblea pulsed"
