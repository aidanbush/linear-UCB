#include <random>
#include <tuple>
#include <iostream>

#include "environment.h"

RandomEnv::RandomEnv(int numContexts, int numArms, double seed) {
    generator.seed(seed);
    envDistribution = uniform_real_distribution<double>(0.0, 1.0);
    contextDistribution = uniform_int_distribution<int>(0, numContexts -1);

    this->means = torch::zeros({numContexts, numArms});
    this->stdevs = torch::zeros({numContexts, numArms});
    this->bestReward = torch::zeros(numContexts);
    this->timestep = 0;

    this->numContexts = numContexts;
    this->numArms = numArms;

    randomizeEnv();
}

void RandomEnv::randomizeEnv() {
    for (int i = 0; i < numContexts; i++ ){
        for (int j = 0; j < numArms; j++ ){
            this->means[i][j] = envDistribution(generator) * 2;
        }
        this->bestReward[i] = this->means[i].max();
    }

    for (int i = 0; i < numContexts; i++ ){
        for (int j = 0; j < numArms; j++ ){
            this->stdevs[i][j] = envDistribution(generator) / 5;
        }
    }
}

int RandomEnv::genContext() {
    return contextDistribution(generator);
}

torch::Tensor RandomEnv::start() {
    currentContext = genContext();

    torch::Tensor context = torch::zeros(numContexts);
    context[currentContext] = 1;

    return context;
}

tuple<double, torch::Tensor, double> RandomEnv::step(int action) {
    timestep++;

    if (timestep % 15000 == 0 && timestep <= 75000) {
        randomizeEnv();
    }

    double mean = means[currentContext][action].item().to<double>();
    double stdev = stdevs[currentContext][action].item().to<double>();

    normal_distribution<double> rewardDistribution = normal_distribution<double>(mean, stdev);
    double reward = rewardDistribution(generator);

    currentContext = genContext();

    torch::Tensor context = torch::zeros(numContexts);
    context[currentContext] = 1;

    return tuple<double, torch::Tensor, double>{reward, context, bestReward[currentContext].item().to<double>()};
}
