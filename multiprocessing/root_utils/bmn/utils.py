import pandas as pd #что-то было про оптимизацию
import numpy as np #что-то было про оптимизацию
import os

from tqdm import tqdm
from typing import (
    Union,
    Tuple
)
from ROOT import (#переписан на C
    TTree,
    TFile
)
from  itertools import chain
import uproot
from multiprocessing import Process


def trackparams2pandas(f,tree: str,branch: str, fname: str) -> pd.DataFrame:
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
    
    df = pd.DataFrame(data=np.array(lst).transpose(), columns = columns)\
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
    #dd = dask.dataframe.from_pandas(df, npartitions=11)
    writer(df, fname)

def mc2pandas(tree: TTree, attr_name: str, fname: str) -> pd.DataFrame:
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
    #dd = dask.dataframe.from_pandas(df, npartitions=10)
    writer(df, fname)

def hits2pandas(tree: TTree, attr_name: str, fname: str) -> pd.DataFrame:
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
    #dd = dask.dataframe.from_pandas(df, npartitions=6)
    writer(df, fname)

def writer(df: pd.DataFrame, fname: str, encoding: str = 'utf-8', 
             sep: str = '\t') -> None:
    print(f"Save data to the `{fname}``")
    df.to_csv(fname, encoding=encoding, sep=sep)
    print("Complete!")

def root2pandas (fname_out: str, 
                fname_params_out: str,
                fname: str, 
                tree_name: str ='cbmsim', 
                hit_obj_name: str ='BmnGemStripHit', 
                mc_obj_name: str ='StsPoint',
                track_params_obj_name: str ='MCTrack'
                                                        ) -> None: 
    print("Read file '%s'" % fname)
    f = TFile(fname)
    
    tree = f.Get(tree_name)
    
    print("File processing...")
    if tree.GetBranch(mc_obj_name):
       ff = uproot.open(fname)
       p = Process(target=mc2pandas, args=(tree, mc_obj_name, fname_out,))
       #mc2pandas(tree, mc_obj_name, fname_out)
       p1 = Process(target=trackparams2pandas, args=(ff,tree_name,track_params_obj_name, fname_params_out,))
       #trackparams2pandas(ff,tree_name,track_params_obj_name, fname_params_out)
       p.start()
       p1.start()
       p.join()
       p1.join()
       
    # if file with hits
    elif tree.GetBranch(hit_obj_name):
        hits2pandas(tree, hit_obj_name, fname_out)
    else:
        raise ValueError("File format is not supported. None of the branches "
                         f"[{hit_obj_name}, {mc_obj_name}] exists")

    print("Complete!")
    #return result    