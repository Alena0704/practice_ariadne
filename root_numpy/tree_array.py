import ROOT
import pandas as pd 
import numpy as np
from root_numpy import tree2array
f = ROOT.TFile('evetest_CC4GeVmb_110_n50k.root')
tree = f.Get('cbmsim')
tracks_params = tree2array(tree, branches='BmnGemStripHit')