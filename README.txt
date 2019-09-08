# Individual Research Project - Controlling Robots with Neural-Networks - Second Experience - Robot Arm

This repository contains all the code required to run the second experience of Alexandre Chenu's Individual Research Project 
applied to a robot arm. The objective is to learn both high-performing and diverse controllers for a robot arm to reach 
any point in the work space. In this experience, an archive of approximately 1000 Neural Networks controllers is learned 
using a Quality-Diversity algorithm. 

## Dependencies

* Ubuntu
* Eigen 3, http://eigen.tuxfamily.org/
* Boost library
* SFERES2 library, https://github.com/sferes2/sferes2 (qd branch)
* NN2 additional module for SFERES2, https://github.com/sferes2/nn2

## Models

models/ contains several evolved Neural Network based controllers.  

* model_10000.bin is trained to reach most of randomly positioned target

## How to run the demo

1. Clone the repository.

```
git clone https://github.com/AlexandreChenu/exp_sampling.git
```

2. Move to singularity repository.

```
cd exp_sampling/singulariy
```

3. Build the demonstration singularity image.

```
./build_final_image.sh
```

This should create a new executable final_exp_sampling_DATE_TIME.sh.

4. Run the demonstration

```
./final_exp_sampling_DATE_TIME.sh
```

## How to modify code and experiences

All the code contained in this repository can be modified in order to adapt it and continue its development. 
All dependencies are installed and pre-compiled on a singularity container. Therefore, you only need to create a sandbox 
container using the two following commands. 

1. Move to singularity container 
```
cd exp_sampling/singulariy
```

2. Build sandbox container 
```
./start_container.sh
```

After editing the code, you must compile it using file. Please refer to https://gitlab.doc.ic.ac.uk/AIRL/AIRL_WIKI/wikis/how-to-use-AIRL_environment-and-create-you-own-experience
for more information about the environment and how to use it. 

## How to modify singularity containers

Singularity containers may be personalized by editing the singularity.def file. 
```
vim singularity.def
```

## Material

Here is a quick summary of all the main files contained in this repository. 

* wscript - waf script for compilation
* ex_sample - main file for running Quality-Diversity algorithm (contains includes, parameters, template-type declaration..)
* best_fit_nn - redefinition of bestfit.hpp from sferes::dart (saves the best model contained in the archive)
* best_fit_it - redefinition of bestfit.hpp from sferes::dart (saves the n best model contained in the archive, by default n = 3)
* best_fit_samp_div - redefinition of bestfit.hpp from sferes::dart (saves the novelty score of the archive)
* fit_behav_new - definition of the evaluation step for a set of samples already computed (samples_cart.txt or samples.txt)
* fit_behav_stoch - definition of the evaluation step for a set of samples computed on the fly
* gen_mlp - definition of a new Feed-Forward Neural Network genotype
* test_model - testing model for a given sample
* test_model_sample - testing model for N samples
* visu_arm - visualisation of the arm trajectory


## Contributors

* Alexandre Chenu 
* Antoine Cully
* Szymon Brych
