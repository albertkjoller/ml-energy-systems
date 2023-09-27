Machine Learning for Energy Systems
==============================

Repository associated with assignments from the course 46765 Machine Learning for Signal Processing @ DTU (fall 2023).

## Setup
Create a conda environment (preferably named `46765`) and install the requirements from the `requirements.txt` file:
```
conda create -n 46765 python=3.11
conda activate 46765
pip install -r requirements.txt
```

Now, create a folder in the root of the repository named `data` and move all the pre-given material to this folder - then you are ready for running the data processing from `src/assignment1/notebooks/data_processing.ipynb` to get the general dataset.

Assignments overview
------------
### __Assignment 1:__ Renewable energy trading in day-ahead and balancing markets

This assignment deals with investigating the problem of trading renewable energy in day-ahed and balancing markets by comparing two strategies; namely, 1) using predictions of the wind power production before adressing the decision-making problem and 2) directly determining the most effective offering strategy using data-driven methods. In the process, a deterministic optimization model is combined with linear and non-linear data-driven methods that are explored along with the effect of regularization.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
