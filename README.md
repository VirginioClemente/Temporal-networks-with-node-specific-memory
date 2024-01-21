# MaxEnt-For-Temporal-Networks

This repository contains the code for the implementation of the study conducted in the following works: [1] and [2].
The first study concerns the definition of a maximum entropy model with temporal constraints. For further information regarding the model, please refer to the paper [1].
We have implemented various types of models, varying the degree of heterogeneity of the constraints. For each of these models, a Python class has been created. The 'experiments' notebook contains the results of applying these models to a real network.

The data related to the application on the real world example [3] of the first paper, are available upon request at the following link: http://realitycommons.media.mit.edu/realitymining4.html.

The second study is related to the usage of the model developed in [1] to improve the commonity detection tast in time evolving networks. 
The data used for the real world application of this second paper are related to the following work: [4].

The algorithm implemented is a modified verison of [5].




[1] Clemente, Giulio Virginio, Claudio J. Tessone, and Diego Garlaschelli. "Temporal networks with node-specific memory: unbiased inference of transition probabilities, relaxation times and structural breaks." arXiv preprint arXiv:2311.16981 (2023).

[2] TO ADD

[3] Eagle, Nathan, Alex Pentland, and David Lazer. "Inferring friendship network structure by using mobile phone data." Proceedings of the national academy of sciences 106.36 (2009): 15274-15278.

[4] Stehlé, Juliette, et al. "High-resolution measurements of face-to-face contact patterns in a primary school." PloS one 6.8 (2011): e23176.

[5] Clauset, Aaron, Mark EJ Newman, and Cristopher Moore. "Finding community structure in very large networks." Physical review E 70.6 (2004): 066111.
