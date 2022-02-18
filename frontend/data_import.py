import os
import shutil

import pandas as pd
import re
import pickle

from backend.functions import *

axes_scale = 'logy'

rootDir = "./raw_data"
importDir = "./imported_data"
for dirPath, _, fileList in os.walk( rootDir ):
    print( f"Found directory: {dirPath}" )
    for fname in fileList:
        print( f"\t{fname}" )
    
    files = [ file for file in fileList if not file.startswith( "." ) and not file.endswith( ".pkl" ) ]
    
    for i, f in enumerate( files ):
        p = re.compile( '\d*[.]?\d*V' )
        voltages = p.findall( f )
        
        dirName = os.path.basename( dirPath )
        f_path = dirPath + "/" + f
        try:
            os.makedirs( f"{importDir}/plots/{dirName}" )
        except:
            pass
        try:
            os.makedirs( f"{importDir}/pickles/{dirName}" )
        except:
            pass
        try:
            os.makedirs( f"{importDir}/data/{dirName}" )
        except:
            pass
        
        if not os.path.splitext( f_path )[ 1 ] == ".dat" or f.startswith( "DG" ):
            pass
        elif os.path.splitext( f_path )[ 1 ] == ".dat":
            df = pd.read_csv( f_path, delim_whitespace=True )
            df.columns = [ "t", "V", "I" ]
            plot_name = f"-{voltages[ 1 ]}"
        elif f.startswith( "DG" ):
            df = pd.read_csv( f_path )
            try:
                df.drop( columns="absI", inplace=True )
            except:
                pass
            df.columns = [ "V", "I" ]
            df[ "t" ] = df.index
            plot_name = f"-{voltages[ 2 ]}"
        else:
            df = pd.read_csv( f_path, index_col=0 )
            df.drop( columns=df.columns[ 4 ], inplace=True )
            df.columns = [ "t", "V", "I", "R" ]
            df.reset_index( inplace=True )
            df.drop( columns="Item", inplace=True )
            plot_name = f"-{voltages[ 2 ]}"
        
        df[ "t" ] = df[ "t" ] - df[ "t" ][ 0 ]
        
        with open( f"{importDir}/pickles/{dirName}/{plot_name}_{i}.pkl", "wb" ) as file:
            pickle.dump( df, file )
        
        fig, _, _ = plot_memristor( df[ "V" ], df[ "I" ], df[ "t" ], plot_name, axes_scale=axes_scale )
        fig.savefig( f"{importDir}/plots/{dirName}/{plot_name}_{i}.png" )
        fig.show()
        
        shutil.copyfile( f_path, f"{importDir}/data/{dirName}/{plot_name}_{i}.csv" )
