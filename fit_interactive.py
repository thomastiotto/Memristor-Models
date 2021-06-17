import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from block_timer.timer import Timer
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from progressbar import progressbar
import scipy.stats as stats
from order_of_magnitude import order_of_magnitude
import os
import multiprocessing as mp
import argparse

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from functions import *


# TODO mouse clicks to set parameters?
# TODO make I-V equations modular
#####################################################
#                         MODEL
#####################################################


class Model():
    def __init__( self, time, voltage ):
        self.V = Interpolated( time, voltage )
        
        self.functions = { "MIMD"    : self.mim_iv,
                           "Ohmic"   : self.ohmic_iv,
                           "Schottky": self.schottky_iv }
        self.set_h1( "MIMD", "MIMD" )
        self.set_h2( "MIMD", "MIMD" )
    
    def g( self, v, Ap, An, Vp, Vn ):
        return np.select( [ v > Vp, v < -Vn ],
                          [ Ap * (np.exp( v ) - np.exp( Vp )),
                            -An * (np.exp( -v ) - np.exp( Vn )) ],
                          default=0 )
    
    def wp( self, x, xp ):
        return ((xp - x) / (1 - xp)) + 1
    
    def wn( self, x, xn ):
        return x / (1 - xn)
    
    def f( self, v, x, xp, xn, alphap, alphan, eta ):
        return np.select( [ eta * v > 0, eta * v <= 0 ],
                          [ np.select( [ x >= xp, x < xp ],
                                       [ np.exp( -alphap * (x - xp) ) * self.wp( x, xp ),
                                         1 ] ),
                            np.select( [ x <= 1 - xn, x > 1 - xn ],
                                       [ np.exp( alphan * (x + xn - 1) ) * self.wn( x, xn ),
                                         1 ] )
                            ] )
    
    def dxdt( self, t, x, Ap, An, Vp, Vn, xp, xn, alphap, alphan, eta=1 ):
        v = self.V( t )
        return eta * self.g( v, Ap, An, Vp, Vn ) * self.f( v, x, xp, xn, alphap, alphan, eta )
    
    def ohmic_iv( self, v, g ):
        return g * v
    
    def mim_iv( self, v, g, b ):
        return g * np.sinh( b * v )
    
    def schottky_iv( v, g, b ):
        return g * (1 - np.exp( -b * v ))
    
    def mim_mim_iv( self, v, gp, bp, gn, bn ):
        return np.piecewise( v, [ v < 0, v >= 0 ],
                             [ lambda v: mim_iv( v, gn, bn ), lambda v: mim_iv( v, gp, bp ) ] )
    
    def I_mim_mim( self, t, x, gmax, bmax, gmin, bmin ):
        v = self.V( t )
        return mim_iv( v, gmax, bmax ) * x + mim_iv( v, gmin, bmin ) * (1 - x)
    
    def I( self, t, x, on_pars, off_pars ):
        v = self.V( t )
        return self.h1( v, *on_pars ) * x + self.h2( v, *off_pars ) * (1 - x)
    
    def set_h1( self, func, func_neg=None ):
        if func_neg:
            self.h1 = lambda v, gp, bp, gn, bn: np.piecewise( v, [ v < 0, v >= 0 ],
                                                              [ lambda v: self.functions[ func ]( v, gn, bn ),
                                                                lambda v: self.functions[ func_neg ]( v, gp, bp ) ] )
        else:
            self.h1 = lambda v, g, b: self.functions[ func ]( v, g, b )
    
    def set_h2( self, func, func_neg=None ):
        if func_neg:
            self.h2 = lambda v, gp, bp, gn, bn: np.piecewise( v, [ v < 0, v >= 0 ],
                                                              [ lambda v: self.functions[ func ]( v, gn, bn ),
                                                                lambda v: self.functions[ func_neg ]( v, gp, bp ) ] )
        else:
            self.h2 = lambda v, g, b: self.functions[ func ]( v, g, b )


class FitWindow( tk.Toplevel ):
    def __init__( self, master, xy ):
        super( FitWindow, self ).__init__( master )
        
        self.title( "Fit " )
        self.geometry( f"1200x600+{xy[ 0 ] + 10}+{xy[ 1 ]}" )
        self.resizable( False, False )
        
        # input window setup
        self.rowconfigure( 0, weight=10 )
        self.columnconfigure( 1, weight=10 )
        
        self.input_setup()
        self.output_setup()
        self.initial_plot()
        
        self.protocol( "WM_DELETE_WINDOW", self.on_close )
        # disable/re-enable the button so we can have only one instance
        self.master.fit_button[ "state" ] = "disabled"
    
    def on_close( self ):
        self.master.Vp.set( self.master.Vp_fit.get() )
        self.master.Vn.set( self.master.Vn_fit.get() )
        
        self.master.plot_update( None )
        
        self.master.fit_button[ "state" ] = "normal"
        self.destroy()
    
    def initial_plot( self ):
        v = self.master.voltage
        i = self.master.current
        _, _, on_mask, off_mask = self.master.fit( v, i, self.master.Vp_fit, self.master.Vn_fit )
        
        self.fig, self.axis = plt.subplots( 1, 1 )
        self.axis.scatter( v[ on_mask ],
                           i[ on_mask ],
                           color="b",
                           label=f"On state" )
        self.axis.scatter( v[ on_mask ],
                           mim_mim_iv( v[ on_mask ], *self.master.get_on_pars() ),
                           color="c",
                           label="On fit",
                           s=1 )
        self.axis.scatter( v[ off_mask ],
                           i[ off_mask ],
                           color="r",
                           label=f"Off state" )
        self.axis.scatter( v[ off_mask ],
                           mim_mim_iv( v[ off_mask ], *self.master.get_off_pars() ),
                           color="m",
                           label="Off fit",
                           s=1 )
        self.axis.set_xlabel( "Voltage (V)" )
        self.axis.set_ylabel( "Current" )
        self.fig.legend()
        self.fig.tight_layout()
        
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        canvas = FigureCanvasTkAgg( self.fig, master=self )
        canvas.draw()
        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().grid( column=0, row=0, columnspan=3 )
        self.fig.canvas.draw()
        
        self.update_output()
    
    def plot_update( self, _ ):
        v = self.master.voltage
        i = self.master.current
        popt_on, popt_off, on_mask, off_mask = self.master.fit( v, i, self.master.Vp_fit, self.master.Vn_fit )
        
        self.master.set_on_pars( popt_on )
        self.master.set_off_pars( popt_off )
        
        # remove old lines
        self.axis.clear()
        
        # Plot new graphs
        self.axis.scatter( v[ on_mask ],
                           i[ on_mask ],
                           color="b",
                           label=f"On state" )
        self.axis.scatter( v[ on_mask ],
                           mim_mim_iv( v[ on_mask ], *self.master.get_on_pars() ),
                           color="c",
                           label="On fit",
                           s=1 )
        self.axis.scatter( v[ off_mask ],
                           i[ off_mask ],
                           color="r",
                           label=f"Off state" )
        self.axis.scatter( v[ off_mask ],
                           mim_mim_iv( v[ off_mask ], *self.master.get_off_pars() ),
                           color="m",
                           label="Off fit",
                           s=1 )
        
        self.fig.canvas.draw()
    
    # TODO vary the polarity threshold
    def update_output( self ):
        self.on_label.set(
                f"h1 = {self.master.gmax_p.get():.2e} * sinh({self.master.bmax_p.get():.2f} * V(t)), V(t) >= 0  "
                f"|  {self.master.gmax_n.get():.2e} * sinh({self.master.bmax_n.get():.2f} * V(t)), "
                f"V(t) < 0" )
        self.off_label.set(
                f"h2 = {self.master.gmin_p.get():.2e} * sinh({self.master.bmin_p.get():.2f} * V(t)), V(t) >= 0  "
                f"|  {self.master.gmin_n.get():.2e} * sinh({self.master.bmin_n.get():.2f} * V(t)), "
                f"V(t) < 0" )
    
    def output_setup( self ):
        self.error = tk.StringVar()
        self.error.set( str( 0.0 ) )
        
        # error_frame = ttk.Frame( self )
        # ttk.Label( error_frame, text="Error" ).grid( column=0, row=0 )
        # ttk.Label( error_frame, textvariable=self.error ).grid( column=0, row=1, sticky=tk.W )
        # for widget in error_frame.winfo_children():
        #     widget.grid( padx=0, pady=0 )
        # error_frame.grid( column=2, row=0 )
        #
        
        self.on_label = tk.StringVar()
        self.off_label = tk.StringVar()
        
        self.update_output()
        
        ttk.Label( self, text="Stable ON fit:" ).grid( column=0, row=1 )
        ttk.Label( self, textvariable=self.on_label ).grid( column=1, row=1, sticky=tk.W )
        ttk.Label( self, text="Stable OFF fit:" ).grid( column=0, row=2 )
        ttk.Label( self, textvariable=self.off_label ).grid( column=1, row=2, sticky=tk.W )
        
        ttk.Label( self,
                   text="Select the negative (Vn) and positive (Vp) threshold voltages which will be used to fit the "
                        "h1 and h2 equations.\nThese equations determine the conductivity profile in the Stable ON "
                        "and Stable OFF states.\nYou can change the values by pulling or clicking on the sliders, "
                        "inputting a value directly and pressing Enter, or selecting the entry box and using Up and "
                        "Down arrows." ).grid(
                column=0, row=6, columnspan=3 )
    
    def set_fit_function( self ):
        return
    
    def input_setup( self ):
        # Vp
        Vp_label = ttk.Label( self, text="Vp" )
        Vp_label.grid( column=0, row=3, padx=0, pady=5 )
        
        Vp_slider = ttk.Scale( self, from_=0, to=np.max( self.master.voltage ), variable=self.master.Vp_fit,
                               command=self.master.plot_update )
        Vp_slider.grid( column=1, row=3, padx=0, pady=5, sticky="EW" )
        
        Vp_entry = ttk.Entry( self, textvariable=self.master.Vp_fit )
        Vp_entry.grid( column=2, row=3, padx=0, pady=5 )
        
        Vp_entry.bind( "<Return>", self.master.plot_update )
        Vp_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.Vp_fit, "up" ) )
        Vp_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.Vp_fit, "down" ) )
        
        # Vn
        Vn_label = ttk.Label( self, text="Vn" )
        Vn_label.grid( column=0, row=4, padx=0, pady=5 )
        
        Vn_slider = ttk.Scale( self, from_=0, to=np.abs( np.min( self.master.voltage ) ), variable=self.master.Vn_fit,
                               command=self.master.plot_update )
        Vn_slider.grid( column=1, row=4, padx=0, pady=5, sticky="EW" )
        
        Vn_entry = ttk.Entry( self, textvariable=self.master.Vn_fit )
        Vn_entry.grid( column=2, row=4, padx=0, pady=5 )
        
        # fitting functions
        functions_frame = ttk.Frame( self )
        functions_frame.grid( column=1, row=5 )
        # h1
        ttk.Label( functions_frame, text="ON fit:" ).grid( column=0, row=0 )
        h1_combobox = ttk.Combobox( functions_frame, values=[ "MIMD", "Ohmic", "Schottky" ], state="disabled" )
        h1_combobox.current( 0 )
        h1_combobox.grid( column=1, row=0, padx=0, pady=5 )
        h1_combobox.bind( "<<ComboboxSelected>>", self.set_fit_function )
        
        # h2
        ttk.Label( functions_frame, text="OFF fit:" ).grid( column=2, row=0 )
        h2_combobox = ttk.Combobox( functions_frame, values=[ "MIMD", "Ohmic", "Schottky" ], state="disabled" )
        h2_combobox.current( 0 )
        h2_combobox.grid( column=3, row=0, padx=0, pady=5 )
        h2_combobox.bind( "<<ComboboxSelected>>", self.set_fit_function )
        
        Vn_entry.bind( "<Return>", self.master.plot_update )
        Vn_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.Vn_fit, "up" ) )
        Vn_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.Vn_fit, "down" ) )


class PlotWindow( tk.Toplevel ):
    def __init__( self, master, xy ):
        super( PlotWindow, self ).__init__( master )
        
        self.title( "Simulated memristor" )
        self.geometry( f"1200x1100+{xy[ 0 ] + 10}+{xy[ 1 ]}" )
        self.resizable( False, False )
        
        self.input_setup()
        self.output_setup()
        self.initial_plot()
        self.plot_data()
        
        self.protocol( "WM_DELETE_WINDOW", self.on_close )
        # disable/re-enable the button so we can have only one instance
        self.master.plot_button[ "state" ] = "disabled"
    
    def on_close( self ):
        self.master.plot_button[ "state" ] = "normal"
        self.destroy()
    
    def update_output( self ):
        self.on_label.set(
                f"h1 = {self.master.gmax_p.get():.2e} * sinh({self.master.bmax_p.get():.2f} * V(t)), V(t) >= 0  "
                f"|  {self.master.gmax_n.get():.2e} * sinh({self.master.bmax_n.get():.2f} * V(t)), "
                f"V(t) < 0" )
        self.off_label.set(
                f"h2 = {self.master.gmin_p.get():.2e} * sinh({self.master.bmin_p.get():.2f} * V(t)), V(t) >= 0  "
                f"|  {self.master.gmin_n.get():.2e} * sinh({self.master.bmin_n.get():.2f} * V(t)), "
                f"V(t) < 0" )
    
    def output_setup( self ):
        self.error = tk.StringVar()
        self.error.set( str( 0.0 ) )
        
        # error_frame = ttk.Frame( self )
        # ttk.Label( error_frame, text="Error" ).grid( column=0, row=0 )
        # ttk.Label( error_frame, textvariable=self.error ).grid( column=0, row=1, sticky=tk.W )
        # for widget in error_frame.winfo_children():
        #     widget.grid( padx=0, pady=0 )
        # error_frame.grid( column=2, row=0 )
        #
        
        self.on_label = tk.StringVar()
        self.off_label = tk.StringVar()
        
        self.update_output()
        
        ttk.Label( self, text="On fit" ).grid( column=0, row=1 )
        ttk.Label( self, textvariable=self.on_label ).grid( column=1, row=1, sticky=tk.W )
        ttk.Label( self, text="Off fit" ).grid( column=0, row=2 )
        ttk.Label( self, textvariable=self.off_label ).grid( column=1, row=2, sticky=tk.W )
        
        ttk.Label( self,
                   text="Select the parameters used to simulated the state variable evolution.\nChanging Vp and Vn "
                        "does not affect the fit for h1 and h2.\nYou can change the "
                        "values by pulling or clicking on the sliders, "
                        "inputting a value directly and pressing Enter, or selecting the entry box and using Up and "
                        "Down arrows." ).grid( column=0, row=11, columnspan=3 )
    
    def input_setup( self ):
        # Ap
        Ap_label = ttk.Label( self, text="Ap" )
        Ap_label.grid( column=0, row=3, padx=0, pady=5 )
        
        Ap_slider = ttk.Scale( self, from_=0.0001, to=10, variable=self.master.Ap,
                               command=self.master.plot_update )
        Ap_slider.grid( column=1, row=3, padx=0, pady=5, sticky="EW" )
        
        Ap_entry = ttk.Entry( self, textvariable=self.master.Ap )
        Ap_entry.grid( column=2, row=3, padx=0, pady=5 )
        
        Ap_entry.bind( "<Return>", self.master.plot_update )
        Ap_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.Ap, "up" ) )
        Ap_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.Ap, "down" ) )
        
        # An
        An_label = ttk.Label( self, text="An" )
        An_label.grid( column=0, row=4, padx=0, pady=5 )
        
        An_slider = ttk.Scale( self, from_=0.0001, to=10, variable=self.master.An,
                               command=self.master.plot_update )
        An_slider.grid( column=1, row=4, padx=0, pady=5, sticky="EW" )
        
        An_entry = ttk.Entry( self, textvariable=self.master.An )
        An_entry.grid( column=2, row=4, padx=0, pady=5 )
        
        An_entry.bind( "<Return>", self.master.plot_update )
        An_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.An, "up" ) )
        An_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.An, "down" ) )
        
        # Vp
        Vp_label = ttk.Label( self, text="Vp" )
        Vp_label.grid( column=0, row=5, padx=0, pady=5 )
        
        Vp_slider = ttk.Scale( self, from_=0, to=np.max( self.master.voltage ), variable=self.master.Vp,
                               command=self.master.plot_update )
        Vp_slider.grid( column=1, row=5, padx=0, pady=5, sticky="EW" )
        
        Vp_entry = ttk.Entry( self, textvariable=self.master.Vp )
        Vp_entry.grid( column=2, row=5, padx=0, pady=5 )
        
        Vp_entry.bind( "<Return>", self.master.plot_update )
        Vp_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.Vp, "up" ) )
        Vp_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.Vp, "down" ) )
        
        # Vn
        Vn_label = ttk.Label( self, text="Vn" )
        Vn_label.grid( column=0, row=6, padx=0, pady=5 )
        
        Vn_slider = ttk.Scale( self, from_=0, to=np.abs( np.min( self.master.voltage ) ), variable=self.master.Vn,
                               command=self.master.plot_update )
        Vn_slider.grid( column=1, row=6, padx=0, pady=5, sticky="EW" )
        
        Vn_entry = ttk.Entry( self, textvariable=self.master.Vn )
        Vn_entry.grid( column=2, row=6, padx=0, pady=5 )
        
        Vn_entry.bind( "<Return>", self.master.plot_update )
        Vn_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.Vn, "up" ) )
        Vn_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.Vn, "down" ) )
        
        # xp
        xp_label = ttk.Label( self, text="xp" )
        xp_label.grid( column=0, row=7, padx=0, pady=5 )
        
        xp_slider = ttk.Scale( self, from_=0.0001, to=1, variable=self.master.xp,
                               command=self.master.plot_update )
        xp_slider.grid( column=1, row=7, padx=0, pady=5, sticky="EW" )
        
        xp_entry = ttk.Entry( self, textvariable=self.master.xp )
        xp_entry.grid( column=2, row=7, padx=0, pady=5 )
        
        xp_entry.bind( "<Return>", self.master.plot_update )
        xp_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.xp, "up" ) )
        xp_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.xp, "down" ) )
        
        # xn
        xn_label = ttk.Label( self, text="xn" )
        xn_label.grid( column=0, row=8, padx=0, pady=5 )
        
        xn_slider = ttk.Scale( self, from_=0.0001, to=1, variable=self.master.xn,
                               command=self.master.plot_update )
        xn_slider.grid( column=1, row=8, padx=0, pady=5, sticky="EW" )
        
        xn_entry = ttk.Entry( self, textvariable=self.master.xn )
        xn_entry.grid( column=2, row=8, padx=0, pady=5 )
        
        xn_entry.bind( "<Return>", self.master.plot_update )
        xn_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.xn, "up" ) )
        xn_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.xn, "down" ) )
        
        # alphap
        alphap_label = ttk.Label( self, text="alphap" )
        alphap_label.grid( column=0, row=9, padx=0, pady=5 )
        
        alphap_slider = ttk.Scale( self, from_=0.0001, to=10, variable=self.master.alphap,
                                   command=self.master.plot_update )
        alphap_slider.grid( column=1, row=9, padx=0, pady=5, sticky="EW" )
        
        alphap_entry = ttk.Entry( self, textvariable=self.master.alphap )
        alphap_entry.grid( column=2, row=9, padx=0, pady=5 )
        
        alphap_entry.bind( "<Return>", self.master.plot_update )
        alphap_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.alphap, "up" ) )
        alphap_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.alphap, "down" ) )
        
        # alphan
        alphan_label = ttk.Label( self, text="alphan" )
        alphan_label.grid( column=0, row=10, padx=0, pady=5 )
        
        alphan_slider = ttk.Scale( self, from_=0.0001, to=10, variable=self.master.alphan,
                                   command=self.master.plot_update )
        alphan_slider.grid( column=1, row=10, padx=0, pady=5, sticky="EW" )
        
        alphan_entry = ttk.Entry( self, textvariable=self.master.alphan )
        alphan_entry.grid( column=2, row=10, padx=0, pady=5 )
        
        alphan_entry.bind( "<Return>", self.master.plot_update )
        alphan_entry.bind( '<Up>', lambda e: self.master.nudge_var( self.master.alphan, "up" ) )
        alphan_entry.bind( '<Down>', lambda e: self.master.nudge_var( self.master.alphan, "down" ) )
        
        save_button = ttk.Button( self, text="Save", command=self.save ).grid( column=0, row=12 )
        #     debug
        debug_button = ttk.Button( self, text="Debug", command=self.debug ).grid( column=1, row=12 )
    
    def save( self ):
        parameters = {
                "x0"    : self.master.x0,
                "Ap"    : self.master.Ap.get(),
                "An"    : self.master.An.get(),
                "Vp"    : self.master.Vp.get(),
                "Vn"    : self.master.Vn.get(),
                "xp"    : self.master.xp.get(),
                "xn"    : self.master.xn.get(),
                "alphap": self.master.alphap.get(),
                "alphan": self.master.alphan.get(),
                "gmax_p": self.master.gmax_p.get(),
                "bmax_p": self.master.bmax_p.get(),
                "gmax_n": self.master.gmax_n.get(),
                "bmax_n": self.master.bmax_n.get(),
                "gmin_p": self.master.gmin_p.get(),
                "bmin_p": self.master.bmin_p.get(),
                "gmin_n": self.master.gmin_n.get(),
                "bmin_n": self.master.bmin_n.get()
                }
        try:
            os.mkdir( f"./fitted" )
        except:
            pass
        
        with open( "./fitted/" + os.path.splitext( os.path.basename( self.master.device_file ) )[ 0 ] + ".txt",
                   "w" ) as file:
            file.write( json.dumps( parameters ) )
        with open( "./fitted/" + os.path.splitext( os.path.basename( self.master.device_file ) )[ 0 ] + ".pkl",
                   "wb" ) as file:
            pickle.dump( parameters, file )
        self.fig.savefig( "./fitted/" + os.path.splitext( os.path.basename( self.master.device_file ) )[ 0 ] + ".png" )
    
    def initial_plot( self ):
        # simulate the model
        # x_solve_ivp = solve_ivp( self.master.memristor.dxdt, (self.master.time[ 0 ], self.master.time[ -1 ]),
        #                          [ self.master.x0 ],
        #                          method="LSODA",
        #                          t_eval=self.master.time,
        #                          args=[ self.master.Ap.get(), self.master.An.get(), self.master.Vp.get(),
        #                                 self.master.Vn.get(), self.master.xp.get(), self.master.xn.get() ] )
        #
        # # updated simulation results
        # t = x_solve_ivp.t
        # x = x_solve_ivp.y[ 0, : ]
        
        x = euler_solver( self.master.memristor.dxdt, self.master.time,
                          dt=np.mean( np.diff( self.master.time ) ),
                          iv=self.master.x0,
                          args=self.master.get_sim_pars() )
        t = self.master.time
        v = self.master.memristor.V( t )
        i = self.master.memristor.I( t, x, self.master.get_on_pars(), self.master.get_off_pars() )
        
        # create the model plot
        self.fig, self.lines, self.axes = plot_memristor( v, i, t, "", figsize=(12, 6), iv_arrows=False )
        
        # plot real data
        self.plot_data()
        
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        canvas = FigureCanvasTkAgg( self.fig, master=self )
        canvas.draw()
        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().grid( column=0, row=0, columnspan=3 )
        
        self.fig.canvas.draw()
    
    def plot_data( self ):
        i = self.master.current
        
        self.axes[ 0 ].plot( self.master.time, i, color="g", alpha=0.5 )
        self.axes[ 2 ].plot( self.master.voltage, i, color="g", alpha=0.5 )
    
    def debug( self ):
        x = euler_solver( self.master.memristor.dxdt, self.master.time,
                          dt=np.mean( np.diff( self.master.time ) ),
                          iv=self.master.x0,
                          args=self.master.get_sim_pars() )
        t = self.master.time
        v = self.master.memristor.V( t )
        i = self.master.memristor.I( t, x, self.master.get_on_pars(), self.master.get_off_pars() )
        
        fig_debug, axes_debug = plt.subplots( 5, 1, figsize=(12, 10) )
        axes_debug[ 0 ].plot( t, v )
        axes_debug[ 0 ].set_ylabel( "Voltage" )
        axes_debug[ 1 ].plot( t, i )
        axes_debug[ 1 ].set_ylabel( "Current" )
        axes_debug[ 2 ].plot( t, x )
        axes_debug[ 2 ].set_ylabel( "State Variable" )
        axes_debug[ 3 ].plot( t, self.master.memristor.g( v, self.master.Ap.get(), self.master.An.get(),
                                                          self.master.Vp.get(),
                                                          self.master.Vn.get() ) )
        axes_debug[ 3 ].set_ylabel( "g" )
        axes_debug[ 4 ].plot( t,
                              self.master.memristor.f( v, x, self.master.xp.get(), self.master.xn.get(),
                                                       self.master.alphap.get(),
                                                       self.master.alphan.get(), 1 ) )
        axes_debug[ 4 ].set_ylabel( "f" )
        for ax in axes_debug.ravel():
            ax.set_xlabel( "Time" )
        fig_debug.tight_layout()
        fig_debug.show()
        
        print( f"max(i): {np.max( i )}, min(i): {np.min( i )} " )
        print( f"max(x): {np.max( x )}, min(x): {np.min( x )} " )
    
    def plot_update( self, _ ):
        # simulate the model
        # x_solve_ivp = solve_ivp( self.master.memristor.dxdt, (self.master.time[ 0 ], self.master.time[ -1 ]),
        #                          [ self.master.x0 ],
        #                          method="LSODA",
        #                          t_eval=self.master.time,
        #                          args=[ self.master.Ap.get(), self.master.An.get(), self.master.Vp.get(),
        #                                 self.master.Vn.get(), self.master.xp.get(), self.master.xn.get() ] )
        #
        # # updated simulation results
        # t = x_solve_ivp.t
        # x = x_solve_ivp.y[ 0, : ]
        
        x = euler_solver( self.master.memristor.dxdt, self.master.time,
                          dt=np.mean( np.diff( self.master.time ) ),
                          iv=self.master.x0,
                          args=self.master.get_sim_pars() )
        t = self.master.time
        v = self.master.memristor.V( t )
        i = self.master.memristor.I( t, x, self.master.get_on_pars(), self.master.get_off_pars() )
        
        # remove old lines
        for ax in self.axes:
            ax.clear()
        
        # Plot new graphs
        self.axes[ 0 ].plot( t, i, color="b" )
        self.axes[ 1 ].plot( t, v, color="r" )
        self.axes[ 2 ].plot( v, i, color="b" )
        
        # plot real data
        self.plot_data()
        
        self.axes[ 0 ].set_ylabel( f"Current (A)", color="b" )
        self.axes[ 0 ].set_xlabel( f"Time (s)" )
        self.axes[ 1 ].set_ylabel( 'Voltage (V)', color='r' )
        self.axes[ 2 ].set_ylabel( f"Current (A)" )
        self.axes[ 2 ].set_xlabel( "Voltage (V)" )
        
        self.fig.canvas.draw()


class MainWindow( tk.Tk ):
    def __init__( self ):
        super( MainWindow, self ).__init__()
        
        # dimensions of the main window
        self.title( "Main" )
        self.xy = (200, 150)
        self.geometry( f"{self.xy[ 0 ]}x{self.xy[ 1 ]}+0+0" )
        self.resizable( False, False )
        
        # variables for simulation
        self.x0 = 0.11
        self.Ap = tk.DoubleVar()
        self.An = tk.DoubleVar()
        self.Vp = tk.DoubleVar()
        self.Vn = tk.DoubleVar()
        self.xp = tk.DoubleVar()
        self.xn = tk.DoubleVar()
        self.alphap = tk.DoubleVar()
        self.alphan = tk.DoubleVar()
        
        self.init_variables()
        
        self.input_setup()
        
        self.plot_window = self.fit_window = None
        
        # variables for fitting
        self.Vp_fit = tk.DoubleVar()
        self.Vn_fit = tk.DoubleVar()
        
        # variables for I-V curve
        self.gmax_p = tk.DoubleVar()
        self.bmax_p = tk.DoubleVar()
        self.gmax_n = tk.DoubleVar()
        self.bmax_n = tk.DoubleVar()
        self.gmin_p = tk.DoubleVar()
        self.bmin_p = tk.DoubleVar()
        self.gmin_n = tk.DoubleVar()
        self.bmin_n = tk.DoubleVar()
    
    def fit( self, v, i, Vp, Vn ):
        Vp = Vp.get()
        Vn = Vn.get()
        
        on_mask = ((v > 0) & (np.gradient( v ) < 0)) \
                  | ((v < 0) & (np.gradient( v ) < 0)
                     & (v > -Vn))
        off_mask = ((v < 0) & (np.gradient( v ) > 0)) \
                   | ((v > 0) & (np.gradient( v ) > 0)
                      & (v < Vp))
        
        popt_on, pcov_on = scipy.optimize.curve_fit( self.memristor.h1, v[ on_mask ], i[ on_mask ] )
        popt_off, pcov_off = scipy.optimize.curve_fit( self.memristor.h2, v[ off_mask ], i[ off_mask ] )
        
        return popt_on, popt_off, on_mask, off_mask
    
    def nudge_var( self, var, direction, amount=0.001 ):
        if direction == "up":
            var.set( var.get() + amount )
        if direction == "down":
            var.set( var.get() - amount )
        
        self.plot_update( None )
    
    def get_sim_pars( self ):
        return [ self.Ap.get(), self.An.get(), self.Vp.get(),
                 self.Vn.get(), self.xp.get(), self.xn.get(),
                 self.alphap.get(), self.alphan.get() ]
    
    def get_on_pars( self ):
        on_pars = [ self.gmax_p.get(), self.bmax_p.get(), self.gmax_n.get(), self.bmax_n.get() ]
        
        return on_pars
    
    def set_on_pars( self, on_pars ):
        gmax_p, bmax_p, gmax_n, bmax_n = on_pars
        
        self.gmax_p.set( gmax_p )
        self.bmax_p.set( bmax_p )
        self.gmax_n.set( gmax_n )
        self.bmax_n.set( bmax_n )
    
    def set_off_pars( self, off_pars ):
        gmin_p, bmin_p, gmin_n, bmin_n = off_pars
        
        self.gmin_p.set( gmin_p )
        self.bmin_p.set( bmin_p )
        self.gmin_n.set( gmin_n )
        self.bmin_n.set( bmin_n )
    
    def get_off_pars( self ):
        off_pars = [ self.gmin_p.get(), self.bmin_p.get(), self.gmin_n.get(), self.bmin_n.get() ]
        
        return off_pars
    
    def plot_update( self, _ ):
        try:
            if self.plot_window.state() == "normal":
                self.plot_window.plot_update( self.plot_window )
                self.plot_window.update_output()
        except:
            pass
        try:
            if self.fit_window.state() == "normal":
                self.fit_window.plot_update( self.fit_window )
                self.fit_window.update_output()
        
        except:
            pass
    
    def read_input( self, device_file ):
        with open( device_file, "rb" ) as file:
            df = pickle.load( file )
        
        self.time = np.array( df[ "t" ].to_list() )[ :-1 ]
        self.current = np.array( df[ "I" ].to_list() )[ :-1 ]
        self.voltage = np.array( df[ "V" ].to_list() )[ :-1 ]
    
    def open_fit_window( self ):
        self.fit_window = FitWindow( self, xy=self.xy )
    
    def open_plot_window( self ):
        self.plot_window = PlotWindow( self, xy=self.xy )
    
    def select_file( self ):
        filetypes = (
                ('Pickled files', '*.pkl'),
                ('All files', '*.*')
                )
        
        self.device_file = fd.askopenfilenames(
                title="Select a device measurement file",
                initialdir=".",
                filetypes=filetypes )[ 0 ]
        
        self.fit_button[ "state" ] = "normal"
        self.plot_button[ "state" ] = "normal"
        self.reset_button[ "state" ] = "normal"
        
        self.read_input( self.device_file )
        
        self.memristor = Model( self.time, self.voltage )
        
        popt_on, popt_off, _, _ = self.fit( self.voltage, self.current, self.Vp, self.Vn )
        self.set_on_pars( popt_on )
        self.set_off_pars( popt_off )
        
        self.plot_update( None )
    
    def init_variables( self ):
        self.Ap.set( 1.0 )
        self.An.set( 1.0 )
        self.Vp.set( 0.0 )
        self.Vn.set( 0.0 )
        self.xp.set( 0.1 )
        self.xn.set( 0.1 )
        self.alphap.set( 0.1 )
        self.alphan.set( 0.1 )
    
    def reset( self ):
        self.read_input( self.device_file )
        
        popt_on, popt_off, _, _ = self.fit( self.voltage, self.current, self.Vp, self.Vn )
        self.set_on_pars( popt_on )
        self.set_off_pars( popt_off )
        
        self.init_variables()
        
        self.plot_update( None )
    
    def input_setup( self ):
        self.open_button = ttk.Button( self, text="Load", command=self.select_file )
        self.open_button.pack()
        self.fit_button = ttk.Button( self, text="Fit", command=self.open_fit_window, state="disabled" )
        self.fit_button.pack()
        self.plot_button = ttk.Button( self, text="Plot", command=self.open_plot_window, state="disabled" )
        self.plot_button.pack()
        self.reset_button = ttk.Button( self, text="Reset", command=self.reset, state="disabled" )
        self.reset_button.pack()
        self.quit_button = ttk.Button( self, text="Quit", command=self.destroy )
        self.quit_button.pack()


app = MainWindow()
app.mainloop()