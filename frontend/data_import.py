import os
import shutil
import itertools

import pandas as pd
import re
import pickle

from backend.functions import *

axes_scale = [ 'linear', 'log', 'symlog' ]
remove_noise = [ False, True ]
print_folder_info = False
show_plots = False

p = list( itertools.product( axes_scale, remove_noise ) )

for scale, noise in p:
    if noise:
        noise_name = 'smoothed'
    else:
        noise_name = 'original'
    # print( f'Importing and printing on {scale} scale with {noise_name} data' )
    
    rootDir = "../raw_data"
    importDir = "../imported_data"
    try:
        os.makedirs( f"{importDir}" )
    except:
        pass
    
    # -- count files in subtree for tqdm
    tot = 0
    for root, dirs, fileList in os.walk( rootDir ):
        files = [ file for file in fileList if not file.startswith( "." ) and not file.endswith( ".pkl" ) ]
        tot += len( files )
    
    pbar = tqdm( total=tot, desc=f'Importing and printing on {scale} scale with {noise_name} data' )
    for dirPath, _, fileList in os.walk( rootDir ):
        if print_folder_info:
            print( f"Found directory: {dirPath}" )
            for fname in fileList:
                print( f"\t{fname}" )
        
        files = [ file for file in fileList if not file.startswith( "." ) and not file.endswith( ".pkl" ) ]
        
        for i, f in enumerate( files ):
            p = re.compile( '\d*[.]?\d*V' )
            voltages = p.findall( f )
            
            dirName = os.path.basename( dirPath )
            f_path = dirPath + "/" + f
            if 'large' in dirName or '100' in dirName:
                um_scale = '100 um'
            elif 'medium' in dirName or '32' in dirName:
                um_scale = '32 um'
            elif 'small' in dirName or '10' in dirName:
                um_scale = '10 um'
            else:
                um_scale = 'n/a'
            
            try:
                os.makedirs( f"{importDir}/plots/{scale}_{'smoothed' if noise else 'original'}/{dirName}" )
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
            
            if os.path.splitext( f_path )[ 1 ] == ".dat":
                df = pd.read_csv( f_path, delim_whitespace=True )
                df.columns = [ "t", "V", "I" ]
                file_name = f"-{voltages[ 1 ]}"
            elif f.startswith( "DG" ):
                df = pd.read_csv( f_path )
                try:
                    df.drop( columns="absI", inplace=True )
                except:
                    pass
                df.columns = [ "V", "I" ]
                df[ "t" ] = df.index
                file_name = f"-{voltages[ 2 ]}"
            elif 'PWL' in f:
                pass
            else:
                df = pd.read_csv( f_path, index_col=0 )
                df.drop( columns=df.columns[ 4 ], inplace=True )
                df.columns = [ "t", "V", "I", "R" ]
                df.reset_index( inplace=True )
                df.drop( columns="Item", inplace=True )
                file_name = f"-{voltages[ 2 ]}"
            
            df[ "t" ] = df[ "t" ] - df[ "t" ][ 0 ]
            
            with open( f"{importDir}/pickles/{dirName}/{file_name}_{i}.pkl", "wb" ) as file:
                pickle.dump( df, file )
            
            plot_name = file_name + f' / {um_scale} / {scale} / {noise_name}'
            
            fig, _, _ = plot_memristor( df[ "V" ], df[ "I" ], df[ "t" ], plot_name,
                                        axes_scale=scale, remove_noise=noise )
            fig.savefig(
                    f"{importDir}/plots/{scale}_{'smoothed' if noise else 'original'}/{dirName}/{file_name}_"
                    f"{i}.png" )
            if show_plots:
                fig.show()
            else:
                plt.close( fig )
            
            shutil.copyfile( f_path, f"{importDir}/data/{dirName}/{file_name}_{i}.csv" )
            
            pbar.update( 1 )
    pbar.close()
