# amyloidPredict
This model was trained to predict amyloid formation of peptides from just the peptide sequence.  
It was trained on 6aa, 10aa, and 15aa peptides and is meant to be applied on IDP (intrinsically disordered protein) or IDR (intrinsically disordered region) fragments of those sizes.

## Quick Start 
I recommend creating a new conda environment:  
```bash
conda create -n amyloidPredict python=3.9
conda activate amyloidPredict
pip install fair-esm # install esm (to get embeddings from 3B parameter ESM2 model)
conda install pytorch pandas scikit-learn matplotlib tqdm 
```
After pip install, download the weights of the 3B parameter ESM2 model locally (~5.3GB) by executing this in python:
```python
import esm
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
```
Then you can **extract ESM embeddings** and **predict amyloidogenicity** of a peptide sequence (one-letter codes) with:  
`python predict.py --sequence VQIVYK`  
which will output a score (between 0-1) for each of the four amyloid classification models.

You can score multiple protein/peptide sequences in a fasta file (e.g. example.fasta) like this:  
`python predict.py --sequence example.fasta`  

If you already have ESM embeddings of sequences you can predict amyloidogenicity faster by pointing to the embeddings file (.pt): 
`python predict.py --embeddingsFile example_embeddings_dir/example_IDR_1.pt`  
or by pointing to the directory with all the embeddings files:  
`python predict.py --embeddingsDir example_embeddings_dir`

Extracting ESM embeddings is reasonably fast on my Mac's CPUs, but is much faster on a GPU. Predicting amyloidogenicity should be very fast on CPUs or GPUs. 

The easiest/smartest way to extract ESM embeddings for many files is with the **extract.py** tool, which I altered slightly from the ESM repo. See their original documentation [here](https://github.com/facebookresearch/esm), or do `python extract.py -h` to see how to use it. Make sure to output the mean representations of the embeddings with the `--include mean` flag. 

## How it works
I made a classification model to predict amyloid formation based on some public amyloid datasets: [15aa tau fragment dataset](https://doi.org/10.1038/s41467-024-45429-2), [10aa TANGO dataset](https://doi.org/10.1038/nbt1012), [6aa WALTZ dataset](http://waltzdb.switchlab.org/sequences).  

I trained three separate models on the 3 datasets, and then trained a more general ensemble model on the combined dataset by using the logits of the three models as features.

Each peptide sequence was featurized with ESM embeddings - 2560 embeddings form the 36-layered, 3B parameter model or 5120 embeddings for the 48-layered, 15B parameter model. 
Then I selected features using a logistic regression model with L1 penalty for each model that killed most features while preserving performance.
Then I used this smaller set of features to train a logistic regression model or support vector machine for each dataset, and finally an ensembled model. See **train_3B.py** and **train_15B.py** in "model_development" for training procedure.

## Predicting amyloidogenicity of larger IDRs and full genomes

The accuracy of each model will degrade if you apply it in ways that deviate from its training. For example, the 6aa model will not classify 15aa fragments as well as the 15aa model. The ensembled model is the most general and may work fine on e.g. 18aa fragments, but will likely perform worse on 30aa fragments.

To predict amyloidogenicity for longer peptides, I recommended breaking a longer protein into many short, overlapping fragments of lengths 6aa, 10aa, and 15aa; then getting a prediction for each fragment; finally aggregating these scores into a per-residue score. You can use a workflow like this: 
1. Break protein/peptides into overlapping fragments using `fragment_fasta.py` (in "human_IDR_scanning" directory):
   - `python fragment_fasta.py IDRs.fasta IDRs_6aa_frag_1sw.fasta 6 1`
   - `python fragment_fasta.py IDRs.fasta IDRs_10aa_frag_2sw.fasta 10 2` # larger sliding windows require less computation time and storage space
   - `python fragment_fasta.py IDRs.fasta IDRs_15aa_frag_3sw.fasta 15 3` 
2. Score each fragment: 
   - `python predict.py -s IDRs_6aa_frag_1sw.fasta --classifiers 6aa -o scores_6aa.csv`
   - `python predict.py -s IDRs_10aa_frag_2sw.fasta --classifiers 10aa -o scores_10aa.csv`
   - `python predict.py -s IDRs_15aa_frag_3ws.fasta --classifiers 15aa -o scores_15aa.csv`
3. Aggregate each fragments' scores into a per-residue score:
   - `python get_perResidue_scores.py` ; see contents of "human_IDR_scanning/get_perResidue_scores" for one way to aggregate fragment scores into residue scores.

This workflow can be used to analyze all the IDRs in the human genome with a few hours of computation on a GPU (~15 hrs on my NVIDIA RTX 3900 Ti).  

I got these human IDRs from [this repo](github.com/KULL-Centre/_2023_Tesei_IDRome) from this very nice [paper](https://doi.org/10.1038/s41586-023-07004-5). 
I did some further analysis on amyloidogenicity patterns for various molecular functions and cellular localizations - see "human_IDR_scanning" if you're interested. 

# Acknowledgments
- The developers of ESM
- Tesei, Lindorff-Larsen, et. al. for their work that inspired parts of this, and for having easy-to-use code on their github
- My advisors: Scott Shell & Joan-Emma Shea