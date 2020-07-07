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
from  itertools import chain
import uproot


def trackparams2pandas(f,tree: str,branch: str) -> pd.DataFrame:
    f_1 = f[tree][branch]
    lst=[]
    columns = ['event', 'track','start_x', 'start_y', 'start_z', 'px', 'py', 'pz', 'pt', 'pdg_code', 'parent']
    arr=f_1["MCTrack.fStartX"].array()
    j=0
    a=[]
    b=[]
    ln = len(arr)
    for i in range(0,ln):
       c = len(arr[i])
       k=np.zeros(c, np.int32)
       k[...] = j
       k1 = np.arange(c, dtype=np.int32)
       a.extend(k)
       b.extend(k1)
       j=j+1
    lst.append(list(a))
    lst.append(list(b))
    arr = np.asarray(arr)
    rn=list(chain.from_iterable(arr))
    lst.append(rn)

    arr=f_1["MCTrack.fStartY"].array()
    arr = np.asarray(arr)
    rn=list(chain.from_iterable(arr))
    lst.append(rn)
    
    arr=f_1["MCTrack.fStartZ"].array()
    arr = np.asarray(arr)
    rn=list(chain.from_iterable(arr))
    lst.append(rn)

    arr=f_1["MCTrack.fPx"].array()
    arr = np.asarray(arr)
    rn=list(chain.from_iterable(arr))
    lst.append(rn)

    arr=f_1["MCTrack.fPy"].array()
    arr = np.asarray(arr)
    rn=list(chain.from_iterable(arr))
    lst.append(rn)

    arr=f_1["MCTrack.fPz"].array()
    arr = np.asarray(arr)
    rn=list(chain.from_iterable(arr))
    lst.append(rn)

    arr=f_1["MCTrack.fStartT"].array()
    arr = np.asarray(arr)
    rn=list(chain.from_iterable(arr))
    lst.append(rn)

    arr=f_1["MCTrack.fPdgCode"].array()
    arr = np.asarray(arr)
    rn=list(chain.from_iterable(arr))
    lst.append(rn)

    arr=f_1["MCTrack.fMotherId"].array()
    arr = np.asarray(arr)
    rn=list(chain.from_iterable(arr))
    lst.append(rn)
    df = pd.DataFrame(data=np.array(lst).transpose(), columns = columns)
    return df

def mc2pandas_for(event_id, e, attr_name):
    MCs = []
    for mc in getattr(e, attr_name):
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
    return MCs

def mc2pandas(tree: TTree, attr_name: str) -> pd.DataFrame:
    MCs = []

    for event_id, e in tqdm(enumerate(tree)):
        for mc in getattr(e, attr_name):
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
    # dataframe columns
    columns = ['event', 'track', 'x_in', 'y_in', 'z_in', 
               'x_out', 'y_out', 'z_out', 'station', 'module']

    # create DataFrame and set columns types
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

def hits2pandas(tree: TTree, attr_name: str) -> pd.DataFrame:
    hits = []
    for event_id, e in tqdm(enumerate(tree)):
        for hit in getattr(e, attr_name):
            hits.append([event_id,
                         hit.GetX(),  
                         hit.GetY(), 
                         hit.GetZ(), 
                         hit.GetStation(),
                         hit.GetModule()])
    # dataframe columns
    columns = ['event', 'x', 'y', 'z', 'station', 'module']

    # create DataFrame and set columns types
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
    ff = uproot.open(fname)
    tree = f.Get(tree_name)
    
    print("File processing...")
    if tree.GetBranch(mc_obj_name):
        result = (
            mc2pandas(tree, mc_obj_name),
            trackparams2pandas(ff,tree_name,track_params_obj_name)
            #trackparams2pandas(tree, track_params_obj_name)
            
        )
    # if file with hits
    elif tree.GetBranch(hit_obj_name):
        result = hits2pandas(tree, hit_obj_name)
    else:
        raise ValueError("File format is not supported. None of the branches "
                         f"[{hit_obj_name}, {mc_obj_name}] exists")

    print("Complete!")
    return result    
