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

#include <boost/test/unit_test.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

//#include "exp/examples2/ex_behav_nn.cpp"

#include "/git/sferes2/exp/examples2/fit_behav.hpp"

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

template <typename T>
int run_simu(T & model, int t_max, std::string filename_in, std::string filename_out) { 

    std::cout << "start initialization" << std::endl;

    //init variables
    double _vmax = 1;
  	double _delta_t = 0.1;
  	double _t_max = t_max; //TMax guidé poto
  	Eigen::Vector3d robot_angles;
    Eigen::Vector3d target;
    

    double n_s = 233;

    std::ofstream output_file; //no need for logfile, we don't care about the trajectory
    std::ifstream input_file; 

    Eigen::Vector3d prev_pos; //compute previous position
    Eigen::Vector3d pos_init;

    std::vector<double> zone_exp(3);
    std::vector<double> res(3);

    robot_angles = {0,M_PI,M_PI}; //init everytime at the same place

    double radius;
    double theta;

    input_file.open(filename_in);
    output_file.open(filename_out);

    if (!input_file) {
          std::cout << "Unable to open file " << filename_in;
          exit(1);   // call system to stop
        }
    if (!output_file) {
          std::cout << "Unable to open file " << filename_out;
          exit(1);   // call system to stop
        }

    std::cout << "initialization done" << std::endl;

    for (int s=0; s < n_s; s++){

      std::cout << "new sample " << s << std::endl;

      model.develop();

      std::cout << "model developed" << std::endl;

      std::cout << "model initialized" << std::endl;

      double dist = 0;

      double out;
      input_file >> out;
      target[0] = out;
      input_file >> out;
      target[1] = out;
      target[2] = 0; 

      robot_angles = {0,M_PI,M_PI}; 

      //get gripper's position
      prev_pos = forward_model(robot_angles);
      pos_init = forward_model(robot_angles);

      std::vector<float> inputs(5);
      //std::vector<float> inputs(2);
    	//iterate through time
      for (int t=0; t< _t_max/_delta_t; ++t){
            
            inputs[0] = target[0] - prev_pos[0]; //get side distance to target
            inputs[1] = target[1] - prev_pos[1]; //get front distance to target
  	        inputs[2] = robot_angles[0];
  	        inputs[3] = robot_angles[1];
  	        inputs[4] = robot_angles[2];

            ////DATA GO THROUGH NN
            //model.nn().init(); //init neural network 

            for (int j = 0; j < model.gen().get_depth() + 1; ++j) //In case of FFNN
              model.nn().step(inputs);
            
            Eigen::Vector3d output;
            for (int indx = 0; indx < 3; ++indx){
              output[indx] = 2*(model.nn().get_outf(indx) - 0.5)*_vmax; //Remap to a speed between -v_max and v_max (speed is saturated)
              robot_angles[indx] += output[indx]*_delta_t; //Compute new angles
            }

            //Eigen::Vector3d new_pos;
            prev_pos = forward_model(robot_angles); //remplacer pour ne pas l'appeler deux fois

            if (sqrt(square(target.array() - prev_pos.array()).sum()) < 0.02){
              dist -= sqrt(square(target.array() - prev_pos.array()).sum());}


           else {
              dist -= (log(1+t)) + (sqrt(square(target.array() - prev_pos.array()).sum()));}
          }

      Eigen::Vector3d final_pos; 
      final_pos = forward_model(robot_angles);

      double out_fit;

      if (sqrt(square(target.array() - final_pos.array()).sum()) < 0.02){
        out_fit = 1.0 + dist/500;} // -> 1

      else {
        out_fit = dist/500;} // -> 0

      output_file << out_fit << "\n";

      std::cout << "fitness: " << out_fit << std::endl;
  }

  std::cout << "test done" << std::endl;

  input_file.close();
  output_file.close();

  return 0;
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

	//const std::string filename = "/git/sferes2/exp/ex_data/2019-08-29_17_54_03_1067/final_model_159.bin";
  const std::string filename = "/git/sferes2/exp/ex_data/2019-09-04_22_59_36_112552/model_10000.bin";
  
	std::cout << "model...loading" << std::endl;
	{
	std::ifstream ifs(filename , std::ios::binary);
	ia_t ia(ifs);
  ia >> model;
	}
  std::cout << "model...loaded" << std::endl;

  // model.develop();

	std::string filename_in = "/git/sferes2/exp/ex_data/samples_cart.txt";
  std::string filename_out = "/git/sferes2/exp/ex_data/2019-09-04_22_59_36_112552/samples_out.txt";

	run_simu(model, 10, filename_in, filename_out);
  
  
  std::cout << "test...done" << std::endl;

	}