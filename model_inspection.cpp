//| This file is a part of the sferes2 framework.
//| Copyright 2016, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.

#include <iostream>
#include <Eigen/Core>

#include <sferes/eval/parallel.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/phen/parameters.hpp>
#include <sferes/run.hpp>
#include <sferes/stat/best_fit.hpp>
#include <sferes/stat/qd_container.hpp>
#include <sferes/stat/qd_selection.hpp>
#include <sferes/stat/qd_progress.hpp>
#include <sferes/fit/fit_qd.hpp>
#include <sferes/qd/container/archive.hpp>
#include <sferes/qd/container/sort_based_storage.hpp>
#include <sferes/qd/container/grid.hpp>
#include <sferes/qd/quality_diversity.hpp>
#include <sferes/qd/selector/tournament.hpp>
#include <sferes/qd/selector/uniform.hpp>
#include <sferes/qd/selector/population_based.hpp>
#include <sferes/qd/selector/value_selector.hpp>
#include <boost/test/unit_test.hpp>
#include <modules/nn2/mlp.hpp>
#include <modules/nn2/gen_dnn.hpp>
#include <modules/nn2/phen_dnn.hpp>
#include <modules/nn2/gen_dnn_ff.hpp>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#include <boost/test/unit_test.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

//#include "exp/examples2/ex_behav_nn.cpp"

#include "fit_behav.hpp"

template <typename T>
int get_depth(T & model) { 

	int depth= model.nn().get_depth();
 
	return depth;
}

int main(int argc, char **argv) {

	using namespace sferes;
	using namespace nn;

	typedef nn_mlp<Params> fit_t; 
	typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> weight_t;
  	//typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> bias_t;
  	typedef PfWSum<weight_t> pf_t;
  	typedef AfSigmoidNoBias<> af_t;
  	typedef sferes::gen::DnnFF<Neuron<pf_t, af_t>,  Connection<weight_t>, Params> gen_t; // TODO : change by DnnFF in order to use only feed-forward neural networks
                                                                                       // TODO : change by hyper NN in order to test hyper NEAT 
  	typedef phen::Dnn<gen_t, fit_t, Params> phen_t;
	typedef boost::archive::binary_iarchive ia_t;

	phen_t model; 


	//for (int i = 0; i<10 ; i++){
	const std::string filename = "/git/sferes2/exp/ex_data/2019-08-14_20_27_35_20038/model_9000.bin";
	//const std::string filename = "/git/sferes2/2019-08-13_14_10_56_1323/final_model_" + std::to_string(104+i) + ".bin";


	std::cout << "model...loading" << std::endl;
	{
	std::ifstream ifs(filename , std::ios::binary);
	ia_t ia(ifs);
  ia >> model;
	}
  std::cout << "model...loaded" << std::endl;

  model.develop();
  model.nn().init();

  std::cout << "model informations: \n \n" << std::endl;

  std::cout << "number of inputs: " << model.nn().get_nb_inputs() << std::endl;
  std::cout << "number of outputs: " << model.nn().get_nb_outputs() << std::endl;
  std::cout << "number of connections: " << model.nn().get_nb_connections() << std::endl;
  std::cout << "number of neurons: " << model.nn().get_nb_neurons() << std::endl;

//}
  //std::cout << "number of depth: " << model.nn().get_graph() << std::endl; //pourquoi il n'accepte pas le get depth? 

  //std::cout << "depth of nn: " << get_depth(model) << std::endl;
  
  
  std::cout << "information gathering...done" << std::endl;

  return 0;

	}