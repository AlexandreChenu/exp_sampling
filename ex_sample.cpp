#include <iostream>
#include <Eigen/Core>

#include <sferes/eval/parallel.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/phen/parameters.hpp>
#include <sferes/run.hpp>
#include <sferes/stat/best_fit.hpp>

//change this line according to the best_fit file you wish to use 
//TODO: Move each best_fit to a new branch or merge everything 

//#include "best_fit_it.hpp" //to save the best N models 
#include "best_fit_samp_div.hpp" //to save fitness and diversity results


#include <sferes/stat/qd_container.hpp>
#include <sferes/stat/qd_selection.hpp>
#include <sferes/stat/qd_progress.hpp>


#include <sferes/fit/fit_qd.hpp>
#include <sferes/qd/container/archive.hpp>
//#include <sferes/qd/container/kdtree_storage.hpp>
#include <sferes/qd/container/sort_based_storage.hpp>
#include <sferes/qd/container/grid.hpp>
#include <sferes/qd/quality_diversity.hpp>
#include <sferes/qd/selector/tournament.hpp>
#include <sferes/qd/selector/uniform.hpp>
#include <sferes/qd/selector/population_based.hpp>
#include <sferes/qd/selector/value_selector.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <modules/nn2/mlp.hpp>
#include <modules/nn2/gen_dnn.hpp>
#include <modules/nn2/phen_dnn.hpp>

//#include <modules/nn2/gen_dnn_ff.hpp>
#include "gen_mlp.hpp"

//#include <exp/examples2/phen_arm.hpp>

#include <cmath>
#include <algorithm>
#include <typeinfo>

#include <cstdlib>

//forward model of a robot arm with N arms
Eigen::Vector3d forward_model(Eigen::VectorXd a){
    
    Eigen::VectorXd _l_arm=Eigen::VectorXd::Ones(a.size()+1);
    _l_arm(0)=0;
    _l_arm = _l_arm/_l_arm.sum();

    Eigen::Matrix4d mat=Eigen::Matrix4d::Identity(4,4);

    for(size_t i=0;i<a.size();i++){

      Eigen::Matrix4d submat;
      submat<<cos(a(i)), -sin(a(i)), 0, _l_arm(i), sin(a(i)), cos(a(i)), 0 , 0, 0, 0, 1, 0, 0, 0, 0, 1;
      mat=mat*submat;
    }
    
    Eigen::Matrix4d submat;
    submat<<1, 0, 0, _l_arm(a.size()), 0, 1, 0 , 0, 0, 0, 1, 0, 0, 0, 0, 1;
    mat=mat*submat;
    Eigen::VectorXd v=mat*Eigen::Vector4d(0,0,0,1);

    return v.head(3);
 }


int main(int argc, char **argv) 
{   
    tbb::task_scheduler_init init(32);
    
    using namespace sferes;
    using namespace nn;

    std::cout << "start exp_sample" <<std::endl;

    typedef nn_mlp<Params> fit_t; 

    typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> weight_t;
    //typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> bias_t; //uncomment if bias is used in the activation function
    
    typedef PfWSum<weight_t> pf_t;
    
    //typedef AfTanhNoBias<params::Dummy> af_t; //Tanh without bias
    typedef AfSigmoidNoBias<> af_t; //Sigmoid without bias 
    //typedef AfSigmoidBias<bias_t> af_t; //Sigmoid with bias
    //typedef AfTanhBias<bias_t> af_t; //Tanh with bias
    
    typedef sferes::gen::GenMlp<Neuron<pf_t, af_t>,  Connection<weight_t>, Params> gen_t; // TODO : change by DnnFF in order to use only feed-forward neural networks
                                                                                       // TODO : change by hyper NN in order to test hyper NEAT 
    
    //typedef sferes::gen::Dnn<Neuron<pf_t, af_t>,  Connection<weight_t>, Params> gen_t; //unconstrained NN architecture
    
    typedef phen::Dnn<gen_t, fit_t, Params> phen_t;
    //typedef qd::selector::Uniform<phen_t, Params> select_t; //TODO : test other selector

    typedef qd::selector::getFitness ValueSelect_t;
    typedef qd::selector::Tournament<phen_t, ValueSelect_t, Params> select_t; 

    typedef qd::container::SortBasedStorage< boost::shared_ptr<phen_t> > storage_t; 
    typedef qd::container::Archive<phen_t, storage_t, Params> container_t; 

    //typedef eval::Eval<Params> eval_t; //(useful for debbuging)
    typedef eval::Parallel<Params> eval_t; //parallel eval (faster)
 
    typedef boost::fusion::vector< 
        stat::BestFitSampDiv<phen_t, Params>, 
        //stat::BestFit<phen_t, Params>,
        stat::QdContainer<phen_t, Params>, 
        stat::QdProgress<phen_t, Params> 
        >
        stat_t; 

    typedef modif::Dummy<> modifier_t; //place holder
    
    typedef qd::QualityDiversity<phen_t, eval_t, stat_t, modifier_t, select_t, container_t, Params> qd_t; 
    //typedef qd::MapElites<phen_t, eval_t, stat_t, modifier_t, Params> qd_t;

    qd_t qd;
    run_ea(argc, argv, qd); 

    //quick output of stats
    std::cout<<"best fitness:" << qd.stat<0>().best()->fit().value() << std::endl;
    std::cout<<"archive size:" << qd.stat<1>().archive().size() << std::endl;
    std::cout << "number of connections of best model" << qd.stat<0>().best()->nn().get_nb_connections() << std::endl;
    std::cout << "number of neurons of best model" << qd.stat<0>().best()->nn().get_nb_neurons() << std::endl;
    std::cout << "exp_sample...done" << std::endl;
    return 0;

  }
