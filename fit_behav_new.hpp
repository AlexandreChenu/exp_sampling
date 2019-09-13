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
//#include <sferes/qd/container/kdtree_storage.hpp>
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
    SFERES_CONST size_t min_nb_neurons  = 4;
    SFERES_CONST size_t max_nb_neurons  = 5;
    SFERES_CONST size_t min_nb_conns  = 50;
    SFERES_CONST size_t max_nb_conns  =  101;
    //SFERES_CONST float  max_weight  = 0.0f;
    //SFERES_CONST float  max_bias  = 0.0f;

    SFERES_CONST float m_rate_add_conn  = 1.0f;
    SFERES_CONST float m_rate_del_conn  = 0.1f;
    SFERES_CONST float m_rate_change_conn = 1.0f;
    SFERES_CONST float m_rate_add_neuron  = 1.0f;
    SFERES_CONST float m_rate_del_neuron  = 1.0f;

    SFERES_CONST int io_param_evolving = true;
    //SFERES_CONST init_t init = random_topology;
    SFERES_CONST init_t init = ff;
  };

  struct mlp {
        SFERES_CONST size_t layer_0_size = 10;
        SFERES_CONST size_t layer_1_size = 10;
    };


    struct nov {
      SFERES_CONST size_t deep = 2;
      SFERES_CONST double l = 0.1; // TODO value ??? minimum mean distance to the k-nearest neighbors to add solution to archive 
      SFERES_CONST double k = 25; // TODO right value? number of k-nearest neighbors for novelty score
      SFERES_CONST double eps = 0.1;// TODO right value??
  };

  // TODO: move to a qd::
  struct pop {
      // number of initial random points
      SFERES_CONST size_t init_size = 100; // nbr of randomly initialized solutions 
      SFERES_CONST size_t size = 100; // size of a batch
      SFERES_CONST size_t nb_gen = 10001; // total nbr of generations 
      SFERES_CONST size_t dump_period = 500; 
  };

  struct qd {

      SFERES_CONST size_t dim = 3; 
      SFERES_CONST size_t behav_dim = 3; // dimension of behavior descriptor
      SFERES_ARRAY(size_t, grid_shape, 100, 100, 100);
  };

  struct sample {

      SFERES_CONST size_t n_samples = 233; // nbr of samples
  };
};


FIT_QD(nn_mlp){

  public :
    //Indiv : still do not know what it is 
    //IO : Neural Network Input and Output type
    template <typename Indiv>

      //void eval(Indiv & ind, IO & input, IO & target){ //ind : altered phenotype
      void eval(Indiv & ind){ //ind : altered phenotype

        //std::cout << "EVALUATION NEW" << std::endl;

        Eigen::Vector3d robot_angles; //angles of the robot
        Eigen::Vector3d target; //target
        std::vector<double> fits(Params::sample::n_samples); //vector of the n_samples targets
        std::vector<double> zone_exp(3); //zone exploration
        std::vector<double> res(3); 

        double fit_median; //median of fitness
        Eigen::MatrixXd zones_exp(Params::sample::n_samples, 3); //matrix of zone exploration for the n_samples evaluations
        std::vector<double> bd_medians(3); //geometric median of the behavior descriptor

        Eigen::MatrixXd samples(Params::sample::n_samples,2); //init samples with cluster sampling 

        std::ifstream samples_stream;
        samples_stream.open("/git/sferes2/exp/exp_sampling/samples_cart.txt"); //previously initialized target already saved

        if (!samples_stream) {
          std::cout << "Unable to open file datafile.txt";
          exit(1);   // call system to stop
        }

        for (int s = 0; s < Params::sample::n_samples ; ++s){ //iterate through several random environements

          //init data
          double dist = 0; //initial cumulative distance equals to zero
          robot_angles = {0,M_PI,M_PI}; //init everytime at the same place
          Eigen::Vector3d pos_init = forward_model(robot_angles); //initial position

	        for (int i=0; i< 3; i++){
		        zone_exp[i] = 0;}
          
	        
          double out;

          samples_stream >> out; //sample reading has been tested
          target[0] = out; //get x coordinate of target
          samples_stream >> out; 
          target[1] = out; //get y coordinate of target

          std::vector<float> inputs(5);//input : front distance / side distance / current angle 1 / angle 2 / angle 3

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
            
            for (int j = 0; j < ind.gen().get_depth() + 1; ++j) //In case of FFNN (remove get_depth if using DNN genotype)
              ind.nn().step(inputs);

            Eigen::Vector3d output;
            for (int indx = 0; indx < 3; ++indx){
              output[indx] = 2*(ind.nn().get_outf(indx) - 0.5)*_vmax; //Remap to a speed between -v_max and v_max (speed is saturated)
              //output[indx] = ind.nn().get_outf(indx)*_vmax; //mapping if using tanh
	      robot_angles[indx] += output[indx]*_delta_t; //Compute new angles
            }

            new_pos = forward_model(robot_angles); //get new position of gripper

            res = get_zone(pos_init, target, new_pos); //compute new zone occupation
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
        } //end for all time-steps, cumulative distance computed

        Eigen::Vector3d final_pos; 
        final_pos = forward_model(robot_angles);

        if (sqrt(square(target.array() - final_pos.array()).sum()) < 0.02){ 
          fits[s] = 1.0 + dist/500; // add 1 if successful reaching
        }

        else {
          fits[s] = dist/500; // nothing added otherwise
        }

        zones_exp(s,0) = zone_exp[0]/(_t_max/_delta_t); //TODO: Generalize to n arms
        zones_exp(s,1) = zone_exp[1]/(_t_max/_delta_t);
        zones_exp(s,2) = zone_exp[2]/(_t_max/_delta_t);
        } //end for all samples 

        samples_stream.close(); //close file

        fit_median = median(fits); //computed median fitness

        int index = geometric_median(zones_exp); 

        bd_medians[0] = zones_exp(index,0); //geometric median is approximated 
        bd_medians[1] = zones_exp(index,1); 
        bd_medians[2] = zones_exp(index,2);

        this->_value = fit_median; //negative mean cumulative distance 

        std::vector<double> desc(3); 
        desc = {bd_medians[0], bd_medians[1], bd_medians[2]};

        this->set_desc(desc); //mean usage of each motor
      }

  double median(std::vector<double> &v) //median computation
  {
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
  }

//forward-model for robot arm (with n arm)
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

//zone computation (TODO: upload sketch)
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
    vO2_M_R0[0] = pos[0];
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

//approximation of geometric median 
//output : index of geometric median
double geometric_median (Eigen::MatrixXd samples){
    
    int n_samples = Params::sample::n_samples;
    Eigen::MatrixXd distances(n_samples,n_samples);
    std::vector<double> sum_dists(n_samples);
    
    for (int i=0; i<n_samples; i++){ //get all distances
        for (int j = i; j < n_samples; j++){
            distances(i,j) = sqrt((samples(i,0)-samples(j,0))*(samples(i,0)-samples(j,0)) + (samples(i,1)-samples(j,1))*(samples(i,1)-samples(j,1)) +(samples(i,2)-samples(j,2))*(samples(i,2)-samples(j,2)));
            distances(j,i) = distances(i,j);}
    }
    
    for (int row=0; row<n_samples; row++){ //compute sum of distances
        double sum =0;
        for (int col=0; col<n_samples; col++){
            sum += distances(row,col);
        }
        sum_dists[row] = sum;
    }
    
    double min_sample;
    min_sample = *std::min_element(sum_dists.begin(), sum_dists.end()); //get minimal distance
    
    int ind = 0;
    while(sum_dists[ind] != min_sample){
        ind+=1;
    }
    
    return ind;
}

private:
  double _vmax = 1; //max joint speed
  double _delta_t = 0.1; //time step
  double _t_max = 10; //simlation time 
};
