## Reproducibility

This folder contains the scripts used to reproduce the main technical analyses reported in the revised ACM TIST paper.





Argument mining evaluation

\--------------------------



Run:

python reproducibility/argument\_mining\_evaluation.py

This evaluates the argument-mining predictions against the human-consensus gold annotations for Conditions A and B.



## Statistical analysis

Run:

python reproducibility/statistical\_analysis.py --gold-a data/gold\_annotations/gold\_A.csv --gold-b data/gold\_annotations/gold\_B.csv

This reproduces the statistical comparison of argument structure between Conditions A and B.



## Grounding analysis

Run:

python reproducibility/grounding\_analysis.py data/grounding/blinded\_grounding\_sample\_120.csv --output-dir results/grounding

This reproduces the grounding analysis using the provided grounding assessment data.



## Requirements

Install the required Python packages using:

pip install -r requirements.txt

Run the commands from the main repository folder.

