import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from typing import (
    Union,
    Tuple
)
from ROOT import (
    TTree,
    TFile
)
cimport numpy as np
cimport cython
import root_numpy as rn

def trackparams2pandas(tree : TTree, attr_name : str):
    #cdef int n = 0
    cdef list e = rn.root2array(tree)
    cdef int track_id, event_id
    tracks_params = []
    print("try read tree in list..")
    #for event_id, e in enumerate(tree):
    for event_id, track_id, tparams in enumerate(e,getattr(attr_name)):
        tracks_params.append([
            event_id,
            track_id,
            tparams.GetStartX(),
            tparams.GetStartY(),
            tparams.GetStartZ(),
            tparams.GetPx(),
            tparams.GetPy(),
            tparams.GetPz(),
            tparams.GetPt(),
            tparams.GetPdgCode(),
            tparams.GetMotherId()
        ])
    cdef list columns = ['event', 'track', 'start_x', 'start_y', 'start_z',
               'px', 'py', 'pz', 'pt', 'pdg_code', 'parent']
    print("create dataframe..")
    df = pd.DataFrame(data=tracks_params, columns=columns) \
           .astype({'event': np.int32, 
                    'track': np.int32,
                    'start_x': np.float32,
                    'start_y': np.float32,
                    'start_z': np.float32,
                    'px': np.float32,
                    'py': np.float32,
                    'pz': np.float32,
                    'pt': np.float32,
                    'pdg_code': np.int16,
                    'parent': np.int32})
    
    return df
    
def mc2pandas(tree : TTree, attr_name : str):
    #cdef int n = 0
    cdef int track_id, event_id
    cdef list e = rn.root2array(tree)
    MCs = []

   # MCs = []
    #for event_id, e in tqdm(enumerate(tree)):
    for event_id,mc in enumerate(e,getattr(attr_name)):
        MCs.append([event_id,
                    mc.GetTrackID(), 
                    mc.GetXIn(), 
                    mc.GetYIn(), 
                    mc.GetZIn(), 
                    mc.GetXOut(), 
                    mc.GetYOut(), 
                    mc.GetZOut(), 
                    mc.GetStation(),
                    mc.GetModule()])
    cdef list columns = ['event', 'track', 'x_in', 'y_in', 'z_in',
               'x_out', 'y_out', 'z_out', 'station', 'module']

    df = pd.DataFrame(data=MCs, columns=columns) \
           .astype({'event': np.int32, 
                    'track': np.int32,
                    'x_in': np.float32,
                    'y_in': np.float32,
                    'z_in': np.float32,
                    'x_out': np.float32,
                    'y_out': np.float32,
                    'z_out': np.float32,
                    'station': np.int8,
                    'module': np.int8})
    return df

def hits2pandas(tree: TTree, attr_name : str):
    cdef list e = rn.root2array(tree)
    cdef int track_id, event_id
    hits = []
    #for event_id, e in tqdm(enumerate(tree)):
    for hit in enumerate(e,getattr(attr_name)): 
        hits.append([event_id,
                    hit.GetX(),  
                    hit.GetY(), 
                    hit.GetZ(), 
                    hit.GetStation(),
                    hit.GetModule()])
    cdef list columns = ['event', 'x', 'y', 'z', 'station', 'module']

    df = pd.DataFrame(data=hits, columns=columns) \
           .astype({'event': np.int32, 
                    'x': np.float32,
                    'y': np.float32,
                    'z': np.float32,
                    'station': np.int8,
                    'module': np.int8})
    return df

def root2pandas(fname: str,
                tree_name: str ='cbmsim',
                hit_obj_name: str ='BmnGemStripHit',
                mc_obj_name: str ='StsPoint',
                track_params_obj_name: str ='MCTrack') -> Union[
                                                            pd.DataFrame,
                                                            Tuple[pd.DataFrame, pd.DataFrame]
                                                        ]:
    print("Read file '%s'" % fname)
    f = TFile(fname)
    # read event tree
    tree = f.Get(tree_name)

    print("File processing...")
    # if file with Monte-Carlo points
    if tree.GetBranch(mc_obj_name):
        result = (
            mc2pandas(tree, mc_obj_name), 
            trackparams2pandas(tree, track_params_obj_name)
        )
    # if file with hits
    elif tree.GetBranch(hit_obj_name):
        result = hits2pandas(tree, hit_obj_name)
    else:
        raise ValueError("File format is not supported. None of the branches "
                         f"[{hit_obj_name}, {mc_obj_name}] exists")

    print("Complete!")
    return result    
