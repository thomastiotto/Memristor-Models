import numpy as np

import backend.functions as functions
import backend.models as models


class Experiment():
    def __init__( self, sim_args, model, input_function, memristor_args, input_args, window_function_args=None ):
        self.name = None
        self.t_max = sim_args[ "t_max" ]
        self.frequency = sim_args[ "frequency" ]
        
        self.t_min = 0
        
        self.simulation = {
                "x0": sim_args[ "x0" ]
                }
        self.recalculate_time( self.t_max )
        
        self.input_args = input_args
        self.input_args.update( { "t_max": self.t_max } )
        self.window_function_args = window_function_args
        
        self.input_function = input_function( **self.input_args )
        self.window_function = functions.WindowFunction( **self.window_function_args ) \
            if self.window_function_args else None
        
        self.memristor_args = memristor_args
        self.memristor_args.update( { "x0": sim_args[ "x0" ] } )
        
        self.memristor = model( self.input_function, **self.memristor_args ) \
            if not self.window_function \
            else model( self.input_function, self.window_function, **self.memristor_args )
        
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
        print( f"\tTime range [ {self.t_min}, {self.t_max} ]" )
        print( f"\tSamples {self.simulation[ 'N' ]}" )
        print( f"\tInitial value of state variable {self.simulation[ 'x0' ]}" )
    
    def recalculate_time( self, t_max ):
        # TODO input period and dt (sampling frequency) should be decoupled
        self.dt = 1 / self.frequency
        self.t_max = t_max
        self.simulation[ "dt" ] = self.dt
        self.simulation[ "t_min" ] = 0
        self.simulation[ "t_max" ] = t_max
        self.simulation[ "time" ] = np.arange( self.t_min, t_max + self.dt, self.dt )
        self.simulation[ "N" ] = (self.t_max - self.t_min) * self.frequency
    
    def fit_memristor( self ):
        pass


class hp_labs_sine( Experiment ):
    
    def __init__( self ):
        super( hp_labs_sine, self ).__init__(
                sim_args={ "t_max": 2, "frequency": 100e3, "x0": 0.1 },
                model=models.HPLabs,
                input_function=functions.Sine,
                memristor_args={ "D": 27e-9, "RON": 10e3, "ROFF": 100e3, "muD": 1e-14 },
                input_args={ "frequency": 1, "vp": 1, "vn": 1 },
                window_function_args={ "type": "joglekar", "p": 7, "j": 1 }
                )
        
        self.name = "HP Labs sine"
        self.fitting.update( {
                "bounds": (0, [ 1e-7, 1e4, 1e5, 1e-13 ]),
                "p0"    : None
                } )


class hp_labs_pulsed( Experiment ):
    
    def __init__( self ):
        super( hp_labs_pulsed, self ).__init__(
                sim_args={ "t_max": 8, "frequency": 100e3, "x0": 0.093 },
                model=models.HPLabs,
                input_function=functions.Triangle,
                memristor_args={ "D": 85e-9, "RON": 1e3, "ROFF": 10e3, "muD": 2e-14 },
                input_args={ "frequency": 0.5, "vp": 1, "vn": 1 },
                window_function_args={ "type": "joglekar", "p": 2, "j": 1 }
                )
        
        self.name = "HP Labs pulsed"
        self.fitting.update( {
                "bounds": (0, [ 1e-7, 1e4, 1e5, 1e-13 ]),
                "p0"    : None
                } )


class miao( Experiment ):
    # TODO frequency of simulation (sampling frequency) != frequency of input
    def __init__( self ):
        super( miao, self ).__init__(
                sim_args={ "t_max": 20, "frequency": 100e3, "x0": 0.11 },
                model=models.Yakopcic,
                input_function=functions.Triangle,
                memristor_args={ "a1"    : 0.11,
                                 "a2"    : 0.11,
                                 "b"     : 0.5,
                                 "Ap"    : 7.5,
                                 "An"    : 2,
                                 "Vp"    : 0.09,
                                 "Vn"    : 0.75,
                                 "alphap": 1,
                                 "alphan": 5,
                                 "xp"    : 0.3,
                                 "xn"    : 0.5,
                                 "eta"   : 1
                                 },
                input_args={ "frequency": 1 / 20, "vp": 0.75, "vn": 1.25 },
                )
        
        self.name = "Miao"
        self.fitting.update(
                { "bounds": ([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [ 1, 1, 1, 1e4, 1e4, 1, 1, 1, 1, 1, 1 ]),
                  "p0"    : [ 0.1, 0.1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ] }
                )


class jo( Experiment ):
    # TODO frequency of simulation (sampling frequency) != frequency of input
    def __init__( self ):
        super( jo, self ).__init__(
                sim_args={ "t_max": 20, "frequency": 100e3, "x0": 0.1 },
                model=models.Yakopcic,
                input_function=functions.Triangle,
                memristor_args={ "a1"    : 3.7e-7,
                                 "a2"    : 4.35e-7,
                                 "b"     : 0.7,
                                 "Ap"    : 0.005,
                                 "An"    : 0.08,
                                 "Vp"    : 1.5,
                                 "Vn"    : 0.5,
                                 "alphap": 1.2,
                                 "alphan": 3,
                                 "xp"    : 0.2,
                                 "xn"    : 0.5,
                                 "eta"   : 1
                                 },
                input_args={ "period": 5, "vp": 4, "vn": 2 },
                )
        
        self.name = "Jo"
        self.fitting.update(
                { "bounds": ([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [ 1, 1, 1, 1e4, 1e4, 1, 1, 1, 1, 1, 1 ]),
                  "p0"    : [ 0.1, 0.1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ] }
                )


class oblea_sine( Experiment ):
    # TODO frequency of simulation (sampling frequency) != frequency of input
    def __init__( self ):
        super( oblea_sine, self ).__init__(
                sim_args={ "t_max": 40e-3, "frequency": 100e3, "x0": 0.11 },
                model=models.Yakopcic,
                input_function=functions.Sine,
                memristor_args={ "gmin": 0.17,
                                 "bmin": 0.05,
                                 "gmax": 0.17,
                                 "bmax": 0.05,
                                 "Ap"  : 4000,
                                 "An"  : 4000,
                                 "Vp"  : 0.16,
                                 "Vn"  : 0.15,
                                 "xp"  : 0.3,
                                 "xn"  : 0.5,
                                 "eta" : 1
                                 },
                input_args={ "frequency": 100, "vp": 0.45, "vn": 0.45 },
                )
        
        self.name = "Oblea sine"
        self.fitting.update(
                { "bounds": ([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [ 1, 1, 1, 1e4, 1e4, 1, 1, 1, 1, 1, 1 ]),
                  "p0"    : [ 0.1, 0.1, 0.01, 1000, 1000, 0.1, 0.1, 1, 1, 0.1, 0.1 ] }
                )


class oblea_pulsed( Experiment ):
    def __init__( self ):
        super( oblea_pulsed, self ).__init__(
                sim_args={ "t_max": 50e-3, "frequency": 100e3, "x0": 0.001 },
                model=models.Yakopcic,
                input_function=functions.Triangle,
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
                input_args={ "frequency": 100, "vp": 0.25, "vn": 0.25 },
                )
        
        self.name = "Oblea pulsed"
        self.fitting.update(
                { "bounds": ([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [ 1, 1, 1, 1e4, 1e4, 1, 1, 1, 1, 1, 1 ]),
                  "p0"    : [ 0.1, 0.1, 0.01, 1000, 1000, 0.1, 0.1, 1, 1, 0.1, 0.1 ] }
                )
