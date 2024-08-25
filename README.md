# Identifying Structural Characteristics in Sparse Lottery Ticket Neural Networks

Jordan Bourdeau, Casey Forey

5/4/24

## About

This project reimplements the procedure from the Lottery Ticket Hypothesis paper \[3]
and builds a codebase to run experiments on different variations of the procedure and its
hyperparameters.

### Lottery Ticket Hypothesis

The Lottery Ticket Hypothesis (LTH) states that it is possible to selectively train networks
with unstructured sparsity through iterative magnitude pruning which are able to match or
exceed the performance of the original network in as many or fewer training iterations.

The original paper investigated several network architectures on the MNIST and CIFAR10 datasets
but due to time constraints we have only implemented the experimental configuration for the
LeNet-300-100 architecture on the MNIST dataset.

## Requirements

This project was developed on Python version 3.11.5 and requires the following packages:

- `tensorflow`
- `matplotlib`
- `numpy`
- `multiprocess`

Along with any dependencies pulled in by these packages.

Running:

```
pip install tensorflow matplotlib numpy multiprocess
```

should be able to install all necessary packages. It may be necessary to create a new Python
environment to do this in depending on package versions installed in the exsting one.

## Running

There are a few top-level Jupyter notebooks which are used primarily for testing/developing.
These can be ran in order to get a glimpse of various components, but the main scripts are in
the following locations.

### Experimental Scripts

**src/experiment_scripts**

Experimental scripts are used to run the LTH procedure and create/train/save models at different
steps in the process. These are set up as Python scripts which take a number of configurable
command line arguments and can be ran as follows:

```
python lenet_300_100_iterative_magnitude_pruning.py --args
```

This will spin up a specified number of processes which will parallelize training of models
and save output to the location specified by the user through the arguments.

**Note: As of now there is only one experiment script for LeNet-300-100 architecture using the MNIST dataset.**

A 'Hello World' version of running experiments can be accessed in the top-level `experiments.ipynb` Jupyter
notebook. Console output is hidden by default, but this can be toggled by changing the argument supplied to the
`verbose` argument in the call to the `get_lenet_300_100_parameters` function.

## References

- [Reference Code](https://github.com/arjun-majumdar/Lottery_Ticket_Hypothesis-TensorFlow_2/blob/master/LTH-LeNet_300_100_10_MNIST_Unstructured_Pruning.ipynb)
- [Dual Lottery Ticket Hypothesis Paper](https://arxiv.org/pdf/2203.04248.pdf)
- [Lottery Ticket Hypothesis Paper + Code](https://paperswithcode.com/paper/the-lottery-ticket-hypothesis-finding-sparse)
- [Lottery Ticket Hypothesis at Scale Paper](https://arxiv.org/abs/1903.01611v1)
- [Elastic Lottery Ticket Hypothesis Paper](https://arxiv.org/abs/2103.16547)
- [Structural Similarities Between Lottery Tickets Paper](https://openreview.net/pdf?id=3l9mLzLa0BA)
- [Mask Similarity for Finding Lottery Tickets Paper](https://arxiv.org/pdf/2007.04091.pdf)
- [Structurally Sparse Lottery Tickets](https://par.nsf.gov/servlets/purl/10355587)
