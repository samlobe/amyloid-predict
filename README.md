# amyloidPredict
This model was trained to predict amyloid formation of peptides from just the peptide sequence.  
It was trained mainly on peptides with 6-15 amino acids using 

**Getting Started**  
I recommend creating a new conda environment:  
```bash
conda create -n amyloidPredict python=3.9
conda activate amyloidPredict
pip install fair-esm # install esm (to get embeddings from 3B parameter ESM2 model)
conda install pytorch pandas scikit-learn matplotlib tqdm # you may not need to install pytorch? I had to on my Mac
```
After pip install, download the weights of the 3B parameter ESM2 model locally (~5.3GB) by executing this in python:
```python
import esm
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
```
Then you can predict amyloidogenicity of a peptide sequence with:  
`python predict.py "VQIVYK"`  
which will output a score (between 0-1) for each residue and each fragment.  

You can score multiple protein/peptide sequences in a fasta file (e.g. many_sequences.fasta) like this:  
`python predict.py many_sequences.fasta`  

**How it works**  
I made a classification model to predict amyloid formation based on some public amyloid datasets.  
Each peptide sequence was featurized with ESM embeddings - 2560 embeddings from the 36-layered, 3B parameter model. (The 15B parameter model seemed to be worse at extrapolating to other datasets.)  
Then a logistic regression model was trained with a L1 penalty which selected a subset of the embeddings.  
I ensembled two logistic regression models to create a final model: the two models were trained on two datasets that were deemed high quality, i.e. if learning transferred well to other datasets.  
The final amyloidogenicity score is the probabilites (between 0-1) that a fragment will be labelled an amyloid.  

The model was trained primarily on peptides between 6 to 15 amino acids long, and it's meant to estimate the propensity for multiple peptides/proteins to form β-sheets when stacked parallely and in-register. 

If you input a peptide/protein longer than 15aa, `predict.py` will break it into multiple fragments, score each fragment, and aggregate the scores.  
By default it will use a sliding window of 3aa to get overlapping 15aa fragments.  
For example, a 21aa peptide will be broken down to 3 fragments: 1-15, 4-18, 7-21.  
It will output a score for each residue, and by default a residue will be assigned the highest score for all the fragments it is in.  
There are options to change the fragment length, the sliding window length, and whether the highest score or the average score per residue is outputted when a residue is in multiple fragments - see **Arguments** below.

Please note that the model has limitations in estimating amyloidogenicity for peptides outside this 6-15 amino acid range, but that it likely provides useful information on a protein's tendency to form parallel, in-register β-sheets.  
I expect this model to be informative for annotating proteins that can form pathological amyloids, functional amyloids, and phase-separating proteins (e.g. participating in LLPS).

**Arguments**  
_Required_:  
a peptide sequence (e.g. "VQIVYK")  
or a fasta file with multiple sequences (e.g. example.fasta)

_Optional_:  
`--name` (`-n`): name for output directory (str)
* By default the sequence (or fasta file prefix) will be used for the output directory
* A fasta file input with multiple sequences will have results output into subdirectories based on the labels within the fasta file.

`--fragment-length` (`-frag`): max fragment length (int, default=15aa). 
* Peptides/proteins above this length will be split into multiple fragments   
  
`--sliding-window` (`-sw`): sliding window length used to break down long peptides into multiple fragments (int, default=3aa)  
* A 25aa input with `-frag 15` and `-sw 5` will be broken down into 3 peptides: 1-15, 6-20, 11-25.  

`--scoring` (`-sc`): method to score residues that are in multiple fragments - "avg" or "highest" (default="highest")
* In the above example, residue 11 is in three fragments. With `-sc highest` the higher score will be used to score it.  
With `-sc avg` the three scores will be averaged.  

`--noGPU`: avoids using a GPU to get the ESM embeddings  
* A GPU would extract these embeddings faster, but a CPU should be reasonably fast.