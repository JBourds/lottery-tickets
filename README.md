# Investigating Lottery Ticket Neural Networks

## Checklist

### Preliminary:
- Form specific hypotheses \[X]
    - What metrics are we examining? \[ ]
    - What are we expecting to happen \[ ]
- Write out procedure \[X]
    - Finding lottery tickets \[X]
    - What is "added" from our code to other work \[X]

### Making Models:
- Create neural architecture(s) \[X]
- Code to train model based on input and random seed \[X]
- Code to test mode performance \[X]
- Code to save model to file \[X]
- Code to run lottery ticket procedure \[ ]

### Comparing Models:
- Code for layer-wise comparison? \[ ]
- Code for sign-similarity algorithm? \[ ]

### Visualizations: 

*Break this into more subtasks
- Code to plot results \[ ]

### Project Proposal & Short Video Presentation 

*Preliminary results due

[Template](https://docs.google.com/document/d/1Ibxb8Egomb8bcWKkVDugm8msLasEfDYyuyuCgU5X5Co/edit)
- Introduction \[ ]
- Problem Defintion and Algorithm \[ ]
    - Task definition
    - Dataset
    - Algorithm definition
- Experimental Evaluation \[ ]
    - Methodology
    - Results
    - Discussion
- Related Work \[ ]
- Next Steps \[ ]
    - Elaborate on specific next steps
    - Define roles going forward
- Code and Dataset \[ ]
- Conclusion \[ ]
- Bibliography \[ ]
- Formatting \[ ]
- Video Report \[ ]
    - Problem description (~2 minutes)
    - Dataset description (~2 minutes)
    - Initial results (~2 minutes)
    - Next steps and timeline (~2 minutes)
- Submit \[ ]
    - Project_Proposal_Report_Group_#.pdf
    - Project_Proposal_Video_Group_#.mp4


### Final Report:

[Template](https://docs.google.com/document/d/1afAWQNCTLAsdrjXKEf7ndB2c5xbw5cBnfQUea-eHDr4/edit)
- Introduction \[ ]
- Problem Defintion and Algorithm \[ ]
    - Task definition
    - Algorithm definition
- Experimental Evaluation \[ ]
    - Methodology
    - Results
    - Discussion
- Related Work \[ ]
- Code and Dataset \[ ]
- Conclusion \[ ]
- Bibliography \[ ]
- Formatting \[ ]
- Submit \[ ]
    - Project_Report_Group_#.pdf

### Final Presentation:
- Submit \[ ]
    - Problem description (~3 minutes)
    - Data description (~3 minutes)
    - ML approaches and results (~7 minutes)

### Misc:
- Peer Evaluations \[ ]

## Introduction

The recent trends of deep neural networks becoming increasingly deep stems from the apparent ability of these more sophisticated networks to still generalize well to unseen data. With the ability for enhanced performance, there are a number of drawbacks with continually creating larger networks. Among these are:
1. Increased computational costs to train these networks make it exceedingly prohibitive to conduct research.
2. Huge networks take up a large amount of memory and take a long time to run, making their use prohibitive on resource-constrained technology (old or mobile devices, embedded systems, etc.)
3. It becomes even more difficult to make any sense from the resulting models as the sheer size of them makes them a black box.

With these challenges, techniques for pruning these neural networks and extracting workable subnetworks with fewer parameters has become an active field of research. The techniques explored in this project are derived from the *Lottery Ticket Hypothesis" posed by Frankle and Carbin which states: "A randomly-initialized, dense neural network contains a subnet- work that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations."

In the paper, they employ the following procedure for iteratively extracting winning tickets on both a fully-connected network for the MNIST dataset (LeNet architecture), and on several different convolutional networks for the CIFAR10 dataset (Conv-2, Conv-4, and Conv-6 architectures):

Strategy 1: Iterative pruning with resetting.
1. Randomly initialize a neural network f (x; m ⊙ θ) where θ = θ0 and m = 1|θ| is a mask.
2. Train the network for j iterations, reaching parameters m ⊙ θj .
3. Prune s% of the parameters, creating an updated mask m′ where Pm′ = (Pm − s)%.
4. Reset the weights of the remaining portion of the network to their values in θ0. That is, let θ = θ0.
5. Let m = m′ and repeat steps 2 through 4 until a sufficiently pruned network has been obtained.

For the LeNet and Conv architectures, Pruning is done on a magnitudual layer-wise basis, and connections to outputs are pruned at half of the rate of the rest of the network. For Resnet-18 and VGG-19, pruning is done globally in order to avoid bottlenecking lottery tickets when a layer with few parameters gets overpruned.

Some of the most interesting results seen by these lottery ticket networks is that they:
- Learn faster and hit early stopping earlier during training as they are pruned more (to a certain point)
- Improve performance as opposed to the original network until pruning reaches ~2.1% remaining parameters
- Generalize better, with closer training/test set accuracy

The main explanation for why these networks can be found via this iterative pruning is that with larger networks there is a combinatorial increase in the number of subnetworks, meaning bigger networks are more likely to have a subnetwork which is, by chance, able to match or exceed the performance of the full network. The authors conjecture one possible explanation that SGD selectively seeks out and trains a well-initialized subnetwork. While this implies there are solid odds for discovering lottery ticket networks in a sufficiently large network, it does not examine the reason why the resulting networks are so performant. A crucial step in the procedure for extracting these lottery tickets is to reset the weights of parameters back to their initial, randomly-initialized weights after each round of pruning. Randomly reinitialized weights resulted in the networks which were reset after pruning to learn slower and experience performance dropoffs at much lower levels of pruning. This seems to imply a prerequisite condition for a set of parameters in the original network to form one of these sparse lottery ticket subnetworks which is fulfilled by the original randomly-initialized weights. The purpose of this project is to investigate what exactly these prerequisites may be, and if they are able to be leveraged to more efficiently train for/seek out lottery ticket networks.

## Hypothesis

We hypothesize there are structual similarities across winning lottery tickets from a certain dataset, task, and architecture which are what allow them to be so successful in training sparse networks to match or succeed the performance of unpruned neural networks trained on the same task.

## Procedure

## Results

## Conclusion

## Links
- [Google Drive](https://drive.google.com/drive/folders/1TV3oNUlDSi0IRF1knm-K9NLvfo4Xl9Lr?usp=share_link)

## References
- [Lottery Ticket Hypothesis Paper + Code](https://paperswithcode.com/paper/the-lottery-ticket-hypothesis-finding-sparse)
- [Lottery Ticket Hypothesis at Scale Paper](https://arxiv.org/abs/1903.01611v1)
- [Elastic Lottery Ticket Hypothesis Paper](https://arxiv.org/abs/2103.16547)
- [Structural Similarities Between Lottery Tickets Paper](https://openreview.net/pdf?id=3l9mLzLa0BA)
- [Mask Similarity for Finding Lottery Tickets Paper](https://arxiv.org/pdf/2007.04091.pdf)
- [Structurally Sparse Lottery Tickets](https://par.nsf.gov/servlets/purl/10355587)