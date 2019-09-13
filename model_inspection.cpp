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

//#include <modules/nn2/gen_dnn_ff.hpp>
#include "gen_mlp.hpp"

#include <cmath>
#include <algorithm>
#include <cstdlib>

#include <boost/test/unit_test.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

//#include "exp/examples2/ex_behav_nn.cpp"

#include "fit_behav_new.hpp"

template <typename T>
int get_depth(T & model) { 

	int depth= model.nn().get_depth();
 
	return depth;
}



//Simply output some key information about a saved model 
//Please, check Sferes2 and NN2 documentation if you wish to print other stuff (https://github.com/sferes2/nn2) 
int main(int argc, char **argv) {

	using namespace sferes;
	using namespace nn;

	typedef nn_mlp<Params> fit_t; 
	typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> weight_t;
  	//typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> bias_t;
  	typedef PfWSum<weight_t> pf_t;
  	typedef AfSigmoidNoBias<> af_t;
  	typedef sferes::gen::GenMlp<Neuron<pf_t, af_t>,  Connection<weight_t>, Params> gen_t; // TODO : change by DnnFF in order to use only feed-forward neural networks
                                                                                       // TODO : change by hyper NN in order to test hyper NEAT 
  	typedef phen::Dnn<gen_t, fit_t, Params> phen_t;
	typedef boost::archive::binary_iarchive ia_t;

	phen_t model; 

	const std::string filename = "./ex_sample_2019-08-16_10_25_26_1772/model_5000.bin";

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

  
  std::cout << "information gathering...done" << std::endl;

  return 0;

	}
