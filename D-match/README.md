# D-match: evaluating scoped meaning representations

D-match is a tool that is able to evaluate scoped meaning representations, in this case Discourse Representation Graphs (DRGs). It compares sets of triples and outputs an F-score. The tool can be used to evaluate different DRG-parsers.
It is heavily based on [SMATCH](https://github.com/snowblink14/smatch), with a few modifications. It was developed as part of the [Parallel Meaning Bank](http:/pmb.let.rug.nl).

## Getting Started

```
git clone https://github.com/RikVN/D-match
```

### Prerequisites

D-match runs with Python 2.7. The memory component needs [psutil](https://pypi.python.org/pypi/psutil).

## Differences with SMATCH ##

* D-match takes triples directly as input
* D-match can process multiple DRGs in parallel
* D-match can average over multiple runs of the system
* D-match needs more restarts, since DRGs, on average, have more triples and variables
* D-match can do baseline experiments, comparing a single DRG to a set of DRGs
* D-match can do sense experiments, e.g. ignoring sense or choosing a baseline sense
* D-match can use more smart initial mappings, based on concepts, names, variables order and initial matches
* D-match can print more detailed output, such as specific matching triples and F-scores for each smart mapping
* D-match can have a maximum number of triples for a single DRG, ignoring DRG-pairs with more triples
* D-match can have a memory limit per parallel thread
* Since DRGs have variable types, D-match ensures that different variable types can never match

## Input format ##

DRGs are sets of triples, with each triple on a new line and with DRGs separated by a white line. Lines starting with '#' or '%' are ignored. Individual triples are formatted like this: var1 edge var2. The whitespace separates the values, so it can be spaces or tabs. Everything after the first 3 values is ignored, so it is possible to put a comment there.

A DRG file looks like this:

```
% DRG 1
% Sentence
b3 REF t1
b1 person.n.01 c1 % possible comment
c1 ARG1 x1

% DRG2
% Another sentence
b1 time.n.01 c1
c1 ARG1 t1
b2 Time c2

% DRG3
etc
```

### Variables ###

The names of the variables are important for DRGs, since they denote different types. They always start with a lowercase letter ([a-z]), followed by a number ([\d]+).

The first letter of the variable is used to set the type of the variable:

* b  : DRS label 
* c  : Discourse label
* p  : Propositional referent
* rest: Discourse referent

Note that different types of variables can never match in D-match! 

For more in-depth information about DRSs and DRGs, please see [Basile and Bos (2013)](https://hal.archives-ouvertes.fr/hal-01344582/document) or [Venhuizen (2015)](http://www.rug.nl/research/portal/publications/projection-in-discourse(a83cf9d5-d9ec-4be4-ba8e-e92b7dd4e823).html).

Actual examples of our DRGs can be found in the **data** folder.

## Running D-match

The most vanilla version can be run like this:

```
python d-match.py -f1 FILE1 -f2 FILE2
```

Running with our example data:

```
python d-match.py -f1 data/100_drgs.prod -f2 data/100_drgs.gold
```

### Parameter options ###

```
-f1   : First file with DRG triples, usually produced file (required)
-f2   : Second file with DRG triples, usually gold file (required)
-r    : Number of restarts used (default 100)
-m    : Max number of triples for a DRG to still take them into account - default 0 means no limit
-p    : Number of parallel threads to use (default 1)
-runs : Number of runs to average over, if you want a more reliable result
-s    : What kind of smart initial mapping we use:
       'no'   : No smart mappings
       'order': Smart mapping based on order of the variables in the DRG (b1 maps to b1, b2 to b2, etc)
       'conc' : Smart mapping based on matching concepts (likely to be in the optimal mapping)
       'att'  : Smart mapping based on attribute triples (proper names usually), default
       'init' : Smart mapping based on number of initial matches for a set of mappings
       'freq' : Smart mapping based on frequency of the variables (currently not implemented)
-prin : Print more specific output, such as individual (avg) F-scores for the smart initial mappings, 
        and the matching and non-matching triples
-sense: Use this to do sense experiments
       'normal' : Don't change anything, just use triples as is (default)   
       'wrong'  : Always use the wrong sense for a concept; used to see impact of concept identification
       'ignore' : Ignore sense - meaning we always produced the correct sense
       'base'   : Always use the first sense for a concept (baseline)
-sig  : Number of significant digits to output (default 4)
-b    : Use this for baseline experiments, comparing a single DRG to a list of DRGs. 
        Produced DRG file should contain a single DRG.
-mem  : Memory limit per parallel thread in MBs, default 500        
-ms   : Instead of averaging the score, output a score for each DRG
-pr   : Also output precison and recall instead of only F-score
-v    : Verbose output
-vv   : Very verbose output  
```

### Running some tests ###

Run with 100 restarts, also showing precision and recall,  with 4 parallel threads to speed things up:

```
python d-match.py -f1 data/100_drgs.prod -f2 data/100_drgs.gold -pr -r 100 -p 4
```

Print specific output, while only using the smart mapping based on concepts:

```
python d-match.py -f1 data/100_drgs.prod -f2 data/100_drgs.gold -pr -r 100 -p 4 -prin -s conc
```

Only take smaller DRGs into account, with a maximum of 50 triples:

```
python d-match.py -f1 data/100_drgs.prod -f2 data/100_drgs.gold -pr -r 100 -p 4 -prin -m 50
```

Doing a run that does not care about word sense disambuation:

```
python d-match.py -f1 data/100_drgs.prod -f2 data/100_drgs.gold -pr -r 100 -p 4 -prin -sense ignore
```

Doing a single DRG, printing the matching and non-matching triples:

```
python d-match.py -f1 data/single_drg.prod -f2 data/single_drg.gold -pr -r 100 -p 4 -prin
```

Outputting a score for each DRG (note we use -p 1 to not mess up printing):

```
python d-match.py -f1 data/100_drgs.prod -f2 data/100_drgs.gold -pr -r 100 -p 1 -ms
```

Doing a baseline experiment, comparing a single DRG to a number of DRGs:

```
python d-match.py -f1 data/single_drg.prod -f2 data/100_drgs.gold -pr -r 100 -p 4 -prin --baseline
```

Still doing the baseline experiment, but now we want more reliable scores, so average over 10 runs:

```
python d-match.py -f1 data/single_drg.prod -f2 data/100_drgs.gold -pr -r 100 -p 4 --baseline -runs 10
```

## Application ##

D-match is currently used in the [Parallel Meaning Bank](http://pmb.let.rug.nl/explorer/explore.php) to estimate the semantic similarity between DRGs of different languages. Come check it out!

## Author

* **Rik van Noord** - PhD-student at University of Groningen - [Personal website](http://www.rikvannoord.nl)

## Paper ##

We are currently publishing a paper regarding D-match, once published the reference will be here.

## Acknowledgments

* Thanks to [SMATCH](https://github.com/snowblink14/smatch) for publishing their code open source.
* All members of the [Parallel Meaning Bank](http://pmb.let.rug.nl)
