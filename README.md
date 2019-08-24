# EncDecDRSparsing
The codes to the paper "Discourse Representation Structure Parsing" ACL 2018.

# Requirements

    python 2.7
    pytorch 0.3.0.post4

# Experiments
## data
The data used in the experiments are stored in folder data, the pretrained word embeddings could be got in https://drive.google.com/open?id=1ICyISR-0PhuQYxIsqE5P7_r-OCsETIEU, and then put it into the folder data. 

## train and test
Currently, we do test for each epoch, because the evaluation is carried by external components

    cd EncDecDRSparsing
    mkdir output_dev # storing development outputs
    mkdir output_tst # storing test outputs
    mkdir output_model # storing models
    python encdec.py 
    
## Evaluation

Tree-like structure should be converted into Discourse Representation Graph (DRG) for evaluation by drs2tuple.py. Take output_tst/1.drs for example.
    
    python drs2tuple.py data/test.drs > data/test.tuple
    python drs2tuple.py output_tst/1.drs > output_tst/1.tuple
    python D-match/d-match.py -f1 data/test.tuple output_tst/1.tuple -pr -r 100 -p 10

Note that D-match is implemented in https://github.com/RikVN 

## Demo
The trained model can be got in https://drive.google.com/open?id=1vkhkYt3_Hmtz0x2GuynPT3PZLLu68day (GPU), https://drive.google.com/open?id=1gSkq2KDEtD5dMDYpRtA-B0Iatbc2gPrv (CPU), and then put it into the folder data.

    python sent2drs.py
    
    
  
