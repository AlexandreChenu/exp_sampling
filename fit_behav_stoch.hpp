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
#include <fstream>
#include <Eigen/Core>
#include <random>

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
#include <sferes/qd/container/kdtree_storage.hpp>
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


using namespace sferes;
using namespace sferes::gen::dnn;
using namespace sferes::gen::evo_float;

struct Params {
  struct evo_float {
    SFERES_CONST float mutation_rate = 0.3f;
    SFERES_CONST float cross_rate = 0.1f;
    SFERES_CONST mutation_t mutation_type = polynomial;
    SFERES_CONST cross_over_t cross_over_type = sbx;
    SFERES_CONST float eta_m = 10.0f;
    SFERES_CONST float eta_c = 10.0f;
  };

  struct parameters {
    // maximum value of parameters
    SFERES_CONST float min = -2.0f;
    // minimum value
    SFERES_CONST float max = 2.0f;
  };

  struct dnn {
    SFERES_CONST size_t nb_inputs = 5; // right/left and up/down sensors
    SFERES_CONST size_t nb_outputs  = 3; //usage of each joint
    SFERES_CONST size_t min_nb_neurons  = 15;
    SFERES_CONST size_t max_nb_neurons  = 50;
    SFERES_CONST size_t min_nb_conns  = 20;
    SFERES_CONST size_t max_nb_conns  = 80;
    SFERES_CONST float  max_weight  = 2.0f;
    SFERES_CONST float  max_bias  = 2.0f;

    SFERES_CONST float m_rate_add_conn  = 1.0f;
    SFERES_CONST float m_rate_del_conn  = 1.0f;
    SFERES_CONST float m_rate_change_conn = 1.0f;
    SFERES_CONST float m_rate_add_neuron  = 1.0f;
    SFERES_CONST float m_rate_del_neuron  = 1.0f;

    SFERES_CONST int io_param_evolving = true;
    //SFERES_CONST init_t init = random_topology;
    SFERES_CONST init_t init = ff;
  };

    struct nov {
      SFERES_CONST size_t deep = 2;
      SFERES_CONST double l = 0.08; // TODO value ???
      SFERES_CONST double k = 25; // TODO right value?
      SFERES_CONST double eps = 0.1;// TODO right value??
  };

  // TODO: move to a qd::
  struct pop {
      // number of initial random points
      SFERES_CONST size_t init_size = 100; // nombre d'individus générés aléatoirement 
      SFERES_CONST size_t size = 100; // size of a batch
      SFERES_CONST size_t nb_gen = 10001; // nbr de gen pour laquelle l'algo va tourner 
      SFERES_CONST size_t dump_period = 500; 
  };

  struct qd {

      SFERES_CONST size_t dim = 3;
      SFERES_CONST size_t behav_dim = 3; //taille du behavior descriptor
      SFERES_ARRAY(size_t, grid_shape, 100, 100, 100);
  };

  struct sample {

      SFERES_CONST size_t n_samples = 300; //nombre d'environements aléatoirement générés
      //Eigen::MatrixXd samples = cluster_sampling(100);
  };
};


FIT_QD(nn_mlp){

  public :
    //Indiv : still do not know what it is 
    //IO : Neural Network Input and Output type
    template <typename Indiv>

      //void eval(Indiv & ind, IO & input, IO & target){ //ind : altered phenotype
      void eval(Indiv & ind){ //ind : altered phenotype

        //std::cout << "EVALUATION" << std::endl;

        Eigen::Vector3d robot_angles;
        Eigen::Vector3d target;
        std::vector<double> fits(Params::sample::n_samples);
        std::vector<double> zone_exp(3);
        std::vector<double> res(3);

        // double sum_dist = 0;
        // double mean_dist = 0;
        // Eigen::Vector3d sum_motor_usage;
        double fit_median;
        //std::vector<double> zone1_exp(Params::sample::n_samples);
        //std::vector<double> zone2_exp(Params::sample::n_samples);
        //std::vector<double> zone3_exp(Params::sample::n_samples);
        Eigen::MatrixXd zones_exp(Params::sample::n_samples, 3);
        std::vector<double> bd_medians(3);

        Eigen::MatrixXd samples(Params::sample::n_samples,2); //init samples with cluster sampling 


        for (int s = 0; s < Params::sample::n_samples ; ++s){ //iterate through several random environements

          //init data
          double dist = 0; //initial cumulative distance equals to zero
          robot_angles = {0,M_PI,M_PI}; //init everytime at the same place
          Eigen::Vector3d pos_init = forward_model(robot_angles); //initial position

	        for (int i=0; i< 3; i++)
		        zone_exp[i] = 0;

          //Initialisation random du sample
          double radius = ((double) rand() / (RAND_MAX)); //radius E[0,1]
          double theta = 2*M_PI*(((double) rand() / (RAND_MAX))-0.5);
          target[0] = radius*cos(theta);
          target[1] = radius*sin(theta);

          std::vector<float> inputs(5);//TODO : what input do we use for our Neural network?

          for (int t=0; t< _t_max/_delta_t; ++t){ //iterate through time

            Eigen::Vector3d prev_pos; //compute previous position
            Eigen::Vector3d new_pos;
            prev_pos = forward_model(robot_angles);

            inputs[0] = target[0] - prev_pos[0]; //get side distance to target (-2 < input < 2)
            inputs[1] = target[1] - prev_pos[1]; //get front distance to target (-2 < input < 2)
            inputs[2] = robot_angles[0];
            inputs[3] = robot_angles[1];
            inputs[4] = robot_angles[2];

            //DATA GO THROUGH NN
            ind.nn().init(); //init neural network 
            
            for (int j = 0; j < ind.gen().get_depth() + 1; ++j) //In case of FFNN
              ind.nn().step(inputs);

            Eigen::Vector3d output;
            for (int indx = 0; indx < 3; ++indx){
              output[indx] = 2*(ind.nn().get_outf(indx) - 0.5)*_vmax; //Remap to a speed between -v_max and v_max (speed is saturated)
              robot_angles[indx] += output[indx]*_delta_t; //Compute new angles
            }

            new_pos = forward_model(robot_angles);

            res = get_zone(pos_init, target, new_pos);
            zone_exp[0] = zone_exp[0] + res[0];
            zone_exp[1] = zone_exp[1] + res[1];
            zone_exp[2] = zone_exp[2] + res[2];

            target[2] = 0; //get rid of z coordinate
            new_pos[2] = 0;
	
	         if (sqrt(square(target.array() - new_pos.array()).sum()) < 0.02){
		          dist -= sqrt(square(target.array() - new_pos.array()).sum());
	         }

          else {
              dist -= log(1+t) + sqrt(square(target.array() - new_pos.array()).sum());
          }
        } //end for all time-steps

        Eigen::Vector3d final_pos; 
        final_pos = forward_model(robot_angles);

        if (sqrt(square(target.array() - final_pos.array()).sum()) < 0.02){
          fits[s] = 1.0 + dist/500; // -> 1
        }
        else {
          fits[s] = dist/500; // -> 0
        }

        zones_exp(s,0) = zone_exp[0]/(_t_max/_delta_t); //TODO: Generalize to n arms
        zones_exp(s,1) = zone_exp[1]/(_t_max/_delta_t);
        zones_exp(s,2) = zone_exp[2]/(_t_max/_delta_t);
        } //end for all samples 

        fit_median = median(fits);

        int index = geometric_median(zones_exp);

        bd_medians[0] = zones_exp(index,0); //geometric median is approximated 
        bd_medians[1] = zones_exp(index,1); 
        bd_medians[2] = zones_exp(index,2);

        this->_value = fit_median; //negative mean cumulative distance 

        std::vector<double> desc(3); 
        desc = {bd_medians[0], bd_medians[1], bd_medians[2]};

        this->set_desc(desc); //mean usage of each motor
      }

  double median(std::vector<double> &v)
  {
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
  }

  Eigen::MatrixXd cluster_sampling(int n_s)
  { 
    Eigen::MatrixXd samples(n_s,2);

    double dist = 0;
    double radius = 0;
    double theta = 0;

    for (int i=0; i < n_s/4; ++i){
      radius = 0.25*((double) rand() / (RAND_MAX)); //radius E[0,1]
      theta = 2*M_PI*(((double) rand() / (RAND_MAX))-0.5);
      samples(i,0) = radius*cos(theta);
      samples(i,1) = radius*sin(theta);
    }
    for (int i=n_s/4; i< n_s/2 ; ++i){
      radius = 0.25*((double) rand() / (RAND_MAX)) + 0.25 ; //radius E[0,1]
      theta = 2*M_PI*(((double) rand() / (RAND_MAX))-0.5);
      samples(i,0) = radius*cos(theta);
      samples(i,1) = radius*sin(theta);
    }
    for (int i=n_s/2; i< 3*n_s/4 ; ++i){
      radius = 0.25*((double) rand() / (RAND_MAX)) + 0.5; //radius E[0,1]
      theta = 2*M_PI*(((double) rand() / (RAND_MAX))-0.5);
      samples(i,0) = radius*cos(theta);
      samples(i,1) = radius*sin(theta);
    }
    for (int i=3*n_s/4; i< n_s ; ++i){
      radius = 0.25*((double) rand() / (RAND_MAX)) + 0.75; //radius E[0,1]
      theta = 2*M_PI*(((double) rand() / (RAND_MAX))-0.5);
      samples(i,0) = radius*cos(theta);
      samples(i,1) = radius*sin(theta);
    }
    return samples;
  }

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

std::vector<double> get_zone(Eigen::Vector3d start, Eigen::Vector3d target, Eigen::Vector3d pos){
    
    
    std::vector<double> desc_add (3);
    
    Eigen::Vector3d middle;
    middle[0] = (start[0]+target[0])/2;
    middle[1] = (start[1]+target[1])/2;
    middle[2] = 1;
    
    std::vector<double> distances (3);
    distances = {0,0,0};
    
    distances[0] = sqrt(square(start.array() - pos.array()).sum()); //R1 (cf sketch on page 3)
    distances[1] = sqrt(square(target.array() - pos.array()).sum()); //R2
    distances[2] = sqrt(square(middle.array() - pos.array()).sum()); //d
    
    double D;
    D = *std::min_element(distances.begin(), distances.end()); //get minimal distance
    
    Eigen::Vector3d vO2_M_R0; //vector 02M in frame R0; (cf sketch on page 4)
    //vO2_M_R0[0] = pos[0] - start[0];
    vO2_M_R0[0] = pos[0];
    //vO2_M_R0[1] = pos[1] - start[1];
    vO2_M_R0[1] = pos[1];
    vO2_M_R0[2] = 1;
    
    Eigen::Matrix3d T; //translation matrix
    T << 1,0,-start[0],0,1,-start[1],0,0,1; //translation matrix
    
    Eigen::Vector3d vO2_T;
    vO2_T[0] = target[0] - start[0];
    vO2_T[1] = target[1] - start[1];
    vO2_T[2] = 1;

    double theta = atan2(vO2_T[1], vO2_T[0]) - atan2(1, 0);
    
    if (theta > M_PI){
        theta -= 2*M_PI;
    }
    else if (theta <= -M_PI){
        theta += 2*M_PI;
    }
    
    Eigen::Matrix3d R;
    R << cos(theta), sin(theta), 0, -sin(theta), cos(theta), 0, 0, 0, 1; //rotation matrix
    
    Eigen::Vector3d vO2_M_R1; //vector 02M in frame R1;
    vO2_M_R1 = T*vO2_M_R0;  
    vO2_M_R1 = R*vO2_M_R1;
    
    
    if (vO2_M_R1[0] < 0){ //negative zone (cf sketch on page 3)
        if (D < 0.2) {
            return {-1, 0, 0};
        }
        if (D >= 0.2 && D < 0.4){
            return {0, -1, 0};
        }
        else {
            return {0,0,-1};
        }
    }
    
    else{ //positive zone
        if (D < 0.2) {
            return {1, 0, 0};
        }
        if (D >= 0.2 && D < 0.4){
            return {0, 1, 0};
        }
        else {
            return {0,0,1};
        }
    }
}

double geometric_median (Eigen::MatrixXd samples){
    
    int n_samples = Params::sample::n_samples;
    //int dim = 3;
    Eigen::MatrixXd distances(n_samples,n_samples);
    std::vector<double> sum_dists(n_samples);
    
    for (int i=0; i<n_samples; i++){ //get all distances
        //std::cout << "debug i: " << i << std::endl;
        for (int j = i; j < n_samples; j++){
            //std::cout << "debug j: " << j << std::endl;
            distances(i,j) = sqrt((samples(i,0)-samples(j,0))*(samples(i,0)-samples(j,0)) + (samples(i,1)-samples(j,1))*(samples(i,1)-samples(j,1)) +(samples(i,2)-samples(j,2))*(samples(i,2)-samples(j,2)));
            //std::cout << "debug distance: " << distances(i,j) << std::endl;
            distances(j,i) = distances(i,j);}
    }
    //std::cout << "out distances: " << distances << std::endl;
    
    for (int row=0; row<n_samples; row++){ //compute sum of distances
        double sum =0;
        for (int col=0; col<n_samples; col++){
            sum += distances(row,col);
        }
        sum_dists[row] = sum;
    }
    //std::cout << "out sum distances: " << sum_dists[0] << " " << sum_dists[1] << " " << sum_dists[2] << " " << sum_dists[3] << std::endl;
    
    double min_sample;
    min_sample = *std::min_element(sum_dists.begin(), sum_dists.end()); //get minimal distance
    
    int ind = 0;
    while(sum_dists[ind] != min_sample){
        ind+=1;
    }
    //std::cout << "indice: " << ind << std::endl;
    
    return ind;
}

private:
  //std::default_random_engine generator; 
  //std::uniform_real_distribution<double> distribution(-1.0,1.0);
  double _vmax = 1;
  double _delta_t = 0.1;
  double _t_max = 10; //TMax guidé poto
};
