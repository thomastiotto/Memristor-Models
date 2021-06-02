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

from functions import *
from models import *
from experiments import *

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from functions import *


#####################################################
#                         MODEL
#####################################################

class Model():
    def __init__( self, time, voltage ):
        self.V = Interpolated( time, voltage )
        self.I = self.I_mim_mim_mim_mim
        self.on_par = { "gmaxp": 9.89356358e-05,
                        "bmaxp": 4.95768018e+00,
                        "gmaxn": 1.38215701e-05,
                        "bmaxn": 3.01625878e+00 }
        self.off_par = { "gminp": 1.21787202e-05,
                         "bminp": 7.10131146e+00,
                         "gminn": 4.36419052e-07,
                         "bminn": 2.59501160e+00
                         }
    
    def g( self, v, Ap, An, Vp, Vn ):
        if v > Vp:
            return Ap * (np.exp( v ) - np.exp( Vp ))
        elif v < -Vn:
            return -An * (np.exp( -v ) - np.exp( Vn ))
        else:
            return 0
    
    def wp( self, x, xp ):
        return (xp - x) / (1 - xp) + 1
    
    def wn( self, x, xn ):
        return x / (1 - xn)
    
    def f( self, v, x, xp, xn, eta ):
        if eta * v >= 0:
            if x >= xp:
                return np.exp( -(x - xp) ) * self.wp( x, xp )
            else:
                return 1
        else:
            if x <= 1 - xn:
                return np.exp( (x + xn - 1) ) * self.wn( x, xn )
            else:
                return 1
    
    def ohmic_iv( self, v, g ):
        return g * v
    
    def mim_iv( self, v, g, b ):
        return g * np.sinh( b * v )
    
    def mim_mim_iv( self, v, gp, bp, gn, bn ):
        return np.piecewise( v, [ v < 0, v >= 0 ],
                             [ lambda v: mim_iv( v, gn, bn ), lambda v: mim_iv( v, gp, bp ) ] )
    
    def I_mim_mim( self, t, x, gmax, bmax, gmin, bmin ):
        v = self.V( t )
        return mim_iv( v, gmax, bmax ) * x + mim_iv( v, gmin, bmin ) * (1 - x)
    
    def I_mim_mim_mim_mim( self, t, x ):
        v = self.V( t )
        return mim_mim_iv( v, *list( self.on_par.values() ) ) * x \
               + mim_mim_iv( v, *list( self.off_par.values() ) ) * (1 - x)
    
    def dxdt( self, t, x, Ap, An, Vp, Vn, xp, xn ):
        eta = 1
        v = self.V( t )
        return eta * self.g( v, Ap, An, Vp, Vn ) * self.f( v, x, xp, xn, eta )


class PlotWindow( tk.Tk ):
    def __init__( self ):
        super( PlotWindow, self ).__init__()
        
        self.title( "Simulated memristor" )
        self.geometry( "1200x600+10+30" )
        self.resizable( False, False )
        
        self.read_input()
        
        self.x0 = 0.11
        self.Ap = tk.DoubleVar()
        self.An = tk.DoubleVar()
        self.Vp = tk.DoubleVar()
        self.Vn = tk.DoubleVar()
        self.xp = tk.DoubleVar()
        self.xn = tk.DoubleVar()
        
        self.Ap.set( 1.0 )
        self.An.set( 1.0 )
        self.Vp.set( 0.0 )
        self.Vn.set( 0.0 )
        self.xp.set( 0.1 )
        self.xn.set( 0.1 )
        
        self.memristor = Model( self.time, self.voltage )
        
        self.initial_plot()
        
        parameter_window = ParameterWindow( self )
        parameter_window.grab_set()
    
    def read_input( self ):
        with open( f"./plots/Radius 10 um/-4V_1.pkl", "rb" ) as file:
            df = pickle.load( file )
        
        self.time = np.array( df[ "t" ].to_list() )[ :-1 ]
        self.current = np.array( df[ "I" ].to_list() )[ :-1 ]
        self.resistance = np.array( df[ "R" ].to_list() )[ :-1 ]
        self.conductance = 1 / self.resistance
        self.voltage = np.array( df[ "V" ].to_list() )[ :-1 ]
    
    def initial_plot( self ):
        # simulate the model
        x_solve_ivp = solve_ivp( self.memristor.dxdt, (self.time[ 0 ], self.time[ -1 ]), [ self.x0 ], method="LSODA",
                                 t_eval=self.time,
                                 args=[ self.Ap.get(), self.An.get(), self.Vp.get(), self.Vn.get(), self.xp.get(),
                                        self.xn.get() ] )
        
        # updated simulation results
        t = x_solve_ivp.t
        x = x_solve_ivp.y[ 0, : ]
        v = self.memristor.V( t )
        i = self.memristor.I( t, x )
        
        # create the model plot
        self.fig, self.lines, self.axes = plot_memristor( v, i, t, "Fitted", figsize=(12, 6), iv_arrows=False )
        
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        canvas = FigureCanvasTkAgg( self.fig, master=self )
        canvas.draw()
        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().pack()
        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk( canvas, self )
        toolbar.update()
        # placing the toolbar on the Tkinter window
        canvas.get_tk_widget().pack()
    
    def plot_update( self, *args ):
        # get updated values from GUI
        # Ap_local = Ap.get()
        # An_local = An.get()
        # Vp_local = Vp.get()
        # Vn_local = Vn.get()
        # xp_local = xp.get()
        # xn_local = xn.get()
        
        # print( [ Ap_local, An_local, Vp_local, Vn_local, xp_local, xn_local ] )
        
        # simulate the model
        x_solve_ivp = solve_ivp( self.memristor.dxdt, (self.time[ 0 ], self.time[ -1 ]), [ self.x0 ], method="LSODA",
                                 t_eval=self.time,
                                 args=[ self.Ap.get(), self.An.get(), self.Vp.get(), self.Vn.get(), self.xp.get(),
                                        self.xn.get() ] )
        
        # updated simulation results
        t = x_solve_ivp.t
        x = x_solve_ivp.y[ 0, : ]
        v = self.memristor.V( t )
        i = self.memristor.I( t, x )
        
        # Adjust to new limits
        self.axes[ 0 ].set_xlim( [ 0, np.max( t ) ] )
        self.axes[ 1 ].set_xlim( [ 0, np.max( t ) ] )
        self.axes[ 1 ].set_ylim(
                [ np.min( v ) - np.abs( 0.5 * np.min( v ) ), np.max( v ) + np.abs( 0.5 * np.max( v ) ) ] )
        self.axes[ 2 ].set_xlim(
                [ np.min( v ) - np.abs( 0.5 * np.min( v ) ), np.max( v ) + np.abs( 0.5 * np.max( v ) ) ] )
        
        i_oom = order_of_magnitude.symbol( np.max( i ) )
        i_scaled = i * 1 / i_oom[ 0 ]
        
        # remove old lines
        self.axes[ 0 ].lines.pop( 0 )
        self.axes[ 1 ].lines.pop( 0 )
        self.axes[ 2 ].lines.pop( 0 )
        
        # Plot new graphs
        self.axes[ 0 ].plot( t, i_scaled, color="b" )
        self.axes[ 1 ].plot( t, v, color="r" )
        self.axes[ 2 ].plot( v, i_scaled, color="b" )
        
        # Adjust to new limits
        self.axes[ 0 ].set_ylim( [ np.min( i_scaled ) - np.abs( 0.5 * np.min( i_scaled ) ),
                                   np.max( i_scaled ) + np.abs( 0.5 * np.max( i_scaled ) ) ] )
        self.axes[ 0 ].set_ylabel( f"Current ({i_oom[ 1 ]}A)", color="b" )
        self.axes[ 2 ].set_ylim( [ np.min( i_scaled ) - np.abs( 0.5 * np.min( i_scaled ) ),
                                   np.max( i_scaled ) + np.abs( 0.5 * np.max( i_scaled ) ) ] )
        
        self.fig.canvas.draw()


class ParameterWindow( tk.Toplevel ):
    def __init__( self, master ):
        super( ParameterWindow, self ).__init__( master )
        
        # dimensions of the main window
        self.title( "Model parameters" )
        self.geometry( "1200x250+0+700" )
        self.protocol( "WM_DELETE_WINDOW", self.on_closing )
        self.resizable( False, False )
        
        # input window setup
        self.columnconfigure( 0, weight=1 )
        self.columnconfigure( 1, weight=10 )
        self.columnconfigure( 2, weight=1 )
        
        self.input_setup()
    
    def input_setup( self ):
        # Ap
        Ap_label = ttk.Label( self, text="Ap" )
        Ap_label.grid( column=0, row=0, padx=0, pady=5 )
        
        Ap_slider = ttk.Scale( self, from_=0, to=10, variable=self.master.Ap, command=self.master.plot_update )
        Ap_slider.grid( column=1, row=0, padx=0, pady=5, sticky="EW" )
        
        Ap_entry = ttk.Entry( self, textvariable=self.master.Ap )
        Ap_entry.grid( column=2, row=0, padx=0, pady=5 )
        Ap_entry.bind( "<Return>", self.master.plot_update )
        
        # An
        An_label = ttk.Label( self, text="An" )
        An_label.grid( column=0, row=1, padx=0, pady=5 )
        
        An_slider = ttk.Scale( self, from_=0, to=10, variable=self.master.An, command=self.master.plot_update )
        An_slider.grid( column=1, row=1, padx=0, pady=5, sticky="EW" )
        
        An_entry = ttk.Entry( self, textvariable=self.master.An )
        An_entry.grid( column=2, row=1, padx=0, pady=5 )
        An_entry.bind( "<Return>", self.master.plot_update )
        
        # Vp
        Vp_label = ttk.Label( self, text="Vp" )
        Vp_label.grid( column=0, row=2, padx=0, pady=5 )
        
        Vp_slider = ttk.Scale( self, from_=0, to=4, variable=self.master.Vp, command=self.master.plot_update )
        Vp_slider.grid( column=1, row=2, padx=0, pady=5, sticky="EW" )
        
        Vp_entry = ttk.Entry( self, textvariable=self.master.Vp )
        Vp_entry.grid( column=2, row=2, padx=0, pady=5 )
        Vp_entry.bind( "<Return>", self.master.plot_update )
        
        # Vn
        Vn_label = ttk.Label( self, text="Vn" )
        Vn_label.grid( column=0, row=3, padx=0, pady=5 )
        
        Vn_slider = ttk.Scale( self, from_=0, to=4, variable=self.master.Vn, command=self.master.plot_update )
        Vn_slider.grid( column=1, row=3, padx=0, pady=5, sticky="EW" )
        
        Vn_entry = ttk.Entry( self, textvariable=self.master.Vn )
        Vn_entry.grid( column=2, row=3, padx=0, pady=5 )
        Vn_entry.bind( "<Return>", self.master.plot_update )
        
        # xp
        xp_label = ttk.Label( self, text="xp" )
        xp_label.grid( column=0, row=4, padx=0, pady=5 )
        
        xp_slider = ttk.Scale( self, from_=0, to=1, variable=self.master.xp, command=self.master.plot_update )
        xp_slider.grid( column=1, row=4, padx=0, pady=5, sticky="EW" )
        
        xp_entry = ttk.Entry( self, textvariable=self.master.xp )
        xp_entry.grid( column=2, row=4, padx=0, pady=5 )
        xp_entry.bind( "<Return>", self.master.plot_update )
        
        # xn
        xn_label = ttk.Label( self, text="xn" )
        xn_label.grid( column=0, row=5, padx=0, pady=5 )
        
        xn_slider = ttk.Scale( self, from_=0, to=1, variable=self.master.xn, command=self.master.plot_update )
        xn_slider.grid( column=1, row=5, padx=0, pady=5, sticky="EW" )
        
        xn_entry = ttk.Entry( self, textvariable=self.master.xn )
        xn_entry.grid( column=2, row=5, padx=0, pady=5 )
        xn_entry.bind( "<Return>", self.master.plot_update )
    
    def on_closing( self ):
        pass


# Tkinter windows
app = PlotWindow()
app.mainloop()

# input_window.focus_force()
# input_window.grab_set()
