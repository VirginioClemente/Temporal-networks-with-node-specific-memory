# MaxEnt-For-Temporal-Networks

This repository contains the code for the implementation of the study conducted in [1].
This study concerns the definition of a maximum entropy model with temporal constraints. For further information regarding the model, please refer to the paper [1].
We have implemented various types of models, varying the degree of heterogeneity of the constraints. For each of these models, a Python class has been created. The 'experiments' notebook contains the results of applying these models to a real network.

The files listed above contain the following:

- PersistenceNetFitnesslinks.py: Implementation of the model with memory using dyadic constraints.
- PersistenceNetFitnessGlobal.py: Implementation of the model with memory using global constraint.
- PersistenceNetFitness.py: Implementation of the model with memory using node-level constraints.
- NaiveTCM.py: Implementation of the model without memory using node-level constraints.

In the 'examples' folder, there are three notebooks:
- Test memory.ipynb
- Test no memory.ipynb
- Structural Breaks Memory, Reshuffling, and MSE.ipynb

The first two notebooks contain tests and results reported in the reference paper [1] for models with and without memory. The last notebook, Structural Breaks Memory, Reshuffling, and MSE.ipynb, includes additional results from the main paper as well as the algorithm used for structural breaks detection.


The data related to the application on the real world example are available upon request at the following link: http://realitycommons.media.mit.edu/realitymining4.html.



[1] Clemente, Giulio Virginio, Claudio J. Tessone, and Diego Garlaschelli. "Temporal networks with node-specific memory: unbiased inference of transition probabilities, relaxation times and structural breaks." arXiv preprint arXiv:2311.16981 (2023).
