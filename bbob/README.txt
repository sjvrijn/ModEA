This is the Python version of the COCO experimental framework and the
post-processing tool.

COCO: http://coco.gforge.inria.fr/

Post-processing package: 
  bbob_pproc 

Python version to conduct experiments:  
  exampleexperiment.py   Example script for running a whole COCO experiment.
  exampletiming.py       Example script for running a CPU-timing experiment.
  bbobbenchmarks.py      BBOB benchmark functions.
  fgeneric.py            Data writing (logging) module used in exampleexperiment.py

Where to start:
Check the documentation from http://coco.gforge.inria.fr/

A hint to launch an experiment on a unix-based system:

    nohup nice python exampleexperiment.py > output-file.txt &

  nohup prevents the process to terminate when exiting the terminal
  nice reduces the priority of the process 
  tail output-file.txt then monitors the output