#ifndef BEST_FIT_IT_
#define BEST_FIT_IT_

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <sferes/stat/stat.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/stat/best_fit.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "/git/sferes2/exp/exp_sampling/fit_behav_new.hpp"

namespace sferes {
  namespace stat {
    SFERES_STAT(BestFitIt, Stat) {
    public:
      template<typename E>
      void refresh(const E& ea) {
        assert(!ea.pop().empty());
        _best = *std::max_element(ea.pop().begin(), ea.pop().end(), fit::compare_max());


        this->_create_log_file(ea, "bestfit.dat");
        if (ea.dump_enabled())
          (*this->_log_file) << ea.gen() << " " << ea.nb_evals() << " " << _best->fit().value() << std::endl;

        //change it to depend from params 
        if (_cnt%Params::pop::dump_period == 0){ //for each dump period

          std::vector<boost::shared_ptr<Phen>> pop2(ea.pop()); //create a hard copy of our population
          std::vector<boost::shared_ptr<Phen> > _bests(_nbest); //TODO : check why we cannot set any vector size

          // std::cout << "copied" << std::endl;

          //access the n bests
          for (int i=0; i<_nbest; i++){
            _bests[i] = *std::max_element(pop2.begin(), pop2.end(), fit::compare_max());
            pop2.erase(std::max_element(pop2.begin(), pop2.end(), fit::compare_max())); //TODO: Check if such method is not too expensive otherwise, use template
          }

          // std::cout << "best found" << std::endl;
          pop2.clear();

          typedef boost::archive::binary_oarchive oa_t;

          std::cout << "writing...model" << std::endl;
          const std::string fmodel = std::string("/git/sferes2/exp/tmp/model_") + std::to_string(_cnt) + ".bin";
          {
          std::ofstream ofs(fmodel, std::ios::binary);
          oa_t oa(ofs);
          //oa << model;
          oa << *_best;
          }

          for (int i =0; i<_nbest; i++){
            std::cout << "writing...model..." << std::to_string(i) << std::endl;
            const std::string fmodel = std::string("/git/sferes2/exp/tmp/model_") + std::to_string(_cnt) + std::string("_") + std::to_string(i) + ".bin";
            {
            std::ofstream ofs(fmodel, std::ios::binary);
            oa_t oa(ofs);
            //oa << model;
            oa << *_bests[i];
            }
          }

          std::cout << "models written" << std::endl;}
        _cnt += 1;
      }

      void show(std::ostream& os, size_t k) {
        _best->develop();
        _best->show(os);
        _best->fit().set_mode(fit::mode::view);
        _best->fit().eval(*_best);
      }
      const boost::shared_ptr<Phen> best() const {
        return _best;
      }
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP(_best);
      }

    protected:
      int _cnt = 0; //not sure if it is useful
      boost::shared_ptr<Phen> _best;
      int _nbest = 3;
    };
  }
}
#endif
