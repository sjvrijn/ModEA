#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is written by F.Ye
# To do test


from codes import Config
from functools import partial
from EvolvingES import runCustomizedES

def _runES(iid,ndim,fid,rep,budget):
    storage_file = 'Result'
    representation = [0,0,0,0,0,0,0,0,0,0,0,20,0.25,1,1,1,1,1,1,0.2,0.955,0.5,0,0.3,0.5,2]
    #fitness_result = EvEs.evaluateCustomizedESs(representation,iids=range(Config.ES_num_runs),ndim=Config.experiment_dims,fid=1,budget=Config.GA_budget,num_reps=100,storage_file=storage_file)
    run_data = runCustomizedES(representation=representation,iid=iid,ndim=ndim, fid=fid, rep =rep,budget=budget)



def main():
    _runES(1,10,1,100,Config.GA_budget)
    print("1\n")

if __name__ == '__main__':
        main()
