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

All the relevant code is contained in the `src.assignment1` folder while the results from the associated report can be obtained by running `src.assignment1.notebooks.explainer_notebook.ipynb`. We remark that this takes a while due to 2-layer cross-validation, however, the implementations without cross-validation are fast to run.

### __Assignment 2:__ Real-time control of a battery

This assignment deals with investigating the problem of controlling a battery by learning a policy based on day-ahead prices. Concretely, a Markov Decision Process is specified and solved using the Value Iteration algorithm. From this, the optimal policy is obtained and experiments on the discount factor, reward signal, etc. are conducted by applying the learned policy on a sequence of day-ahead prices.

All the relevant code is contained in the `src.assignment2` folder while the results from the associated report can be obtained by running `src.assignment2.notebooks.explainer_notebook.ipynb`.

### __Assignment 3:__ Day-ahead generation scheduling

This assignment deals with examining the problem of setting the optimal power generation schedule in a day-ahead setting (TSO). The problem is formulated as a linear program and solved using Gurobi after which a data set is extracted from which a machine learning approach to solving the same task is solved.

All the relevant code is contained in the `src.assignment3` folder while the results from the associated report can be obtained by running `src.assignment3.notebooks.explainer_notebook.ipynb`. The extracted data set is located in the `src.assignment3.data` folder. 

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
