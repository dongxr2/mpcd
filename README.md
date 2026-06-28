# Mirror-Paired Contrast Decomposition (MPCD)

This repository contains the raw data and code for the paper **Mirror-Paired Contrast Decomposition: A Symmetry-Aware Estimand Framework for Clustered Action-Outcome Data, with an Application to Professional Tennis** by Guan Wang, Xiaorui Dong, Hanhui Liu, and Jiyan Chen.

## Overview
Repeated categorical actions with binary outcomes are common in observational systems, yet their interpretation depends on whether the target is a typical event or a typical actor. This repository implements **Mirror-Paired Contrast Decomposition (MPCD)**, a symmetry-aware estimand and reporting framework that prespecifies an involution-based reference composition, distinguishes event-average from actor-average behavior, and reports the resulting mirror departure separately from a descriptive difference of within-actor outcome contrasts. 

## Repository contents
Raw data are stored in the split archive `_tennis_cache.part01.rar` / `_tennis_cache.part02.rar` (extract by pointing your tool at part01), and the core analysis code is in the Python (`*.py`) files (including `mpcd_estimand_simulation.py`). Any additional resources will also be released here in the future.

## Requirements
* Python 3.11
* pandas, NumPy, SciPy, statsmodels

```bash
pip install pandas numpy scipy statsmodels
```

## Data sources
Match data are credited to Jeff Sackmann / Tennis Abstract (https://www.tennisabstract.com/) and the Match Charting Project.
