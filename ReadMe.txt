The codes correspond to the paper entitled "Simultaneous Navigation and Radio Mapping for Cellular-Connected UAV with Deep Reinforcement Learning".
A copy of the paper is attached. The paper is also available for download at arXiv.

The codes are implemented with Python + Keras + TensorFlow

The following files are included:

Paper.pdf: The paper that includes all the details about the algorithms.

Dueling_DDQN_MultiStepLearning_main.py: The main program for Dueling DDQN Multi-Step Learning for Coverage-Aware UAV Navigation, corresponding to Aglorithm 1 of the paper.

SNARM_main.py: The main program for SNARM (Simultaneous Navigation and Radio Mapping), corresponding to Algorithm 2 of the paper

radio_environment.py: The program for generating the actual radio environment

radio_mapping.py: The program for defining the RadioMap class

plot_results.py: Plot the moving average return for the two algorithms. 