#include <torch/torch.h>
#include <iostream>
#include <cstdio>
#include <utility>

#include "linUCB.h"
#include "environment.h"

using namespace std;

pair<vector<double>, vector<double>> calcMeanStdev(vector<vector<double>> data) {
    vector<double> means;
    vector<double> stdevs;

    for (int i = 0; i < data[0].size(); i++) {
        double sum = 0;

        for (int j = 0; j < data.size(); j++) {
            sum += data[j][i];
        }

        means.push_back(sum / double(data.size()));

        double stdevTmp = 0;

        for (int j = 0; j < data.size(); j++) {
            stdevTmp += pow(data[j][i] - means[i], 2);
        }

        stdevs.push_back(sqrt(stdevTmp / double(data.size())));
    }

    return pair<vector<double>, vector<double>>{means, stdevs};
}

void printVector(vector<double> vec, FILE *file) {
    for (int i = 0; i < vec.size() -1; i++) {
    //for (double v : vec) {
        fprintf(file, "%f, ", vec[i]);
    }
    fprintf(file, "%f", vec[vec.size()-1]);
}

void printData(pair<vector<double>, vector<double>> data, FILE* file) {
    //print means
    fprintf(file, ": {\n\t\"mean\": [");
    printVector(data.first, file);

    //print stdev
    fprintf(file, "],\n\t\"stdev\": [");
    printVector(data.second, file);

    fprintf(file, "] }\n");
}

void writeResults(vector<vector<double>> returns, vector<vector<double>> regrets) {
    // calculate return mean and stdev
    pair<vector<double>, vector<double>> returnsData = calcMeanStdev(returns);

    pair<vector<double>, vector<double>> regretsData = calcMeanStdev(regrets);

    // print data
    fprintf(stdout, "{\n");
    fprintf(stdout, "\"returns\"");
    printData(returnsData, stdout);
    fprintf(stdout, ", \"regrets\"");
    printData(regretsData, stdout);
    fprintf(stdout, "}\n");
}

pair<vector<double>, vector<double>> runAgent(LinUCB agent, RandomEnv env, int numEpisodes, int episodeLength) {
    vector<double> returns;
    vector<double> regrets;
    double epReturn = 0;
    double epRegret = 0;

    for (int i = 0; i < numEpisodes; i++) {
        fprintf(stderr, "episode %d\n", i);
        epReturn = 0;
        epRegret = 0;

        // run episode
        torch::Tensor context = env.start();
        int action = agent.start(context);

        for (int j = 0; j < episodeLength; j++) { // TODO start at 0 or 1
            tuple<double, torch::Tensor, double> envVals = env.step(action);
            action = agent.step(get<0>(envVals), get<1>(envVals));

            epReturn += get<0>(envVals);
            epRegret += get<2>(envVals) - get<0>(envVals);
            //printf("ret %f, reg %f\n", get<0>(envVals), get<2>(envVals) - get<0>(envVals));
        }

        returns.emplace_back(epReturn);
        regrets.emplace_back(epRegret);
    }

    return pair<vector<double>, vector<double>>{returns, regrets};
}

void run_test(int numRuns, int numEpisodes, int episodeLength, double seed) {
    LinUCB agent = LinUCB(10, 5, 1, .1);
    RandomEnv env = RandomEnv(10, 5, seed);

    // store data
    vector<vector<double>> returns;
    vector<vector<double>> regrets;

    for (int i = 0; i < numRuns; i++) {
        fprintf(stderr, "run: %d\n", i);
        LinUCB agent = LinUCB(10, 5, 1, .1);
        RandomEnv env = RandomEnv(10, 5, seed + i);

        pair<vector<double>, vector<double>> runResults = runAgent(agent, env, numEpisodes, episodeLength);
        returns.push_back(runResults.first);
        regrets.push_back(runResults.second);
    }

    // write to file
    writeResults(returns, regrets);
}

int main() {
    int numRuns = 30;
    int numEpisodes = 12000;
    int episodeLength = 10;
    run_test(numRuns, numEpisodes, episodeLength, 382.4268);

    return 0;
}
