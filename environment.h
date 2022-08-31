#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <random>
#include <tuple>
#include <torch/torch.h>

using namespace std;

class RandomEnv {
    public:
        RandomEnv(int numContexts, int numArms, double seed);

        torch::Tensor start();
        tuple<double, torch::Tensor, double> step(int action);

    private:
        int currentContext;
        int timestep;

        int numContexts;
        int numArms;

        torch::Tensor means;
        torch::Tensor stdevs;

        torch::Tensor bestReward;

        default_random_engine generator;
        uniform_real_distribution<double> envDistribution;
        uniform_int_distribution<> contextDistribution;

        void randomizeEnv();
        int genContext();
};

#endif /* ENVIRONMENT_H */
