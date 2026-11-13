"""Sample-size check for the exploitability lower bound (cfr24, seed 0).
Run from the project root:  python scripts/br_sample_size.py > results/br/size_check.log"""
import os, pickle, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fcd.vgame import *; from fcd.br import *
s=pickle.load(open('results/cfr24.pkl','rb')); cfg,strat=s['config'],s['strategy']
ix=HandIndex(cfg)
FE=deal_features(cfg,300000,2,ix)
FF=deal_features(cfg,1000000,1,ix)
for nfit in (30000,100000,300000,1000000):
  Ff=subset(FF,np.arange(nfit))
  for mode in ('bucket','fine'):
    out=[]
    for b in (0,1):
      vin,pol=best_response(DrawVGame(cfg,Ff,ix,b),strat,b,mode)
      vout,un=evaluate_policy(DrawVGame(cfg,FE,ix,b),strat,b,pol,mode)
      out.append((vin.mean(),vout.mean(),vout.std()/np.sqrt(len(vout)),un))
    ein=(out[0][0]+out[1][0])/2; eout=(out[0][1]+out[1][1])/2; se=np.hypot(out[0][2],out[1][2])/2
    print(nfit,mode,'in %.4f out %.4f se %.4f unseen %.4f %.4f'%(ein,eout,se,out[0][3],out[1][3]),flush=True)
