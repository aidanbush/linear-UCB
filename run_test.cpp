#include <torch/torch.h>
#include <iostream>
#include <cstdio>
#include <utility>

#include <vector>

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
#ifdef SEPARATE_UPDATES
        torch::Tensor prevContext;
        int action;
        int prevAction;

        action = agent.selectAction(context);
        prevAction = action;
        prevContext = context;
#else /* SEPARATE_UPDATES */
        int action = agent.start(context);
#endif /* SEPARATE_UPDATES */

        for (int j = 0; j < episodeLength; j++) { // TODO start at 0 or 1
            // get reward for action, next state, and expected best reward
            tuple<double, torch::Tensor, double> envVals = env.step(action);

#ifdef SEPARATE_UPDATES
            // update agent with previous state and reward
            // select action
            agent.updateAgent(prevContext, prevAction, get<0>(envVals));
            action = agent.selectAction(get<1>(envVals));

            prevAction = action;
            prevContext = get<1>(envVals);
#else /* SEPARATE_UPDATES */
            action = agent.step(get<0>(envVals), get<1>(envVals));
#endif /* SEPARATE_UPDATES */

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
    int num_actions = 4;
    int num_states = 10;
    LinUCB agent = LinUCB(num_states, num_actions, 1, .1, 0);
    RandomEnv env = RandomEnv(num_states, num_actions, seed);

    // store data
    vector<vector<double>> returns;
    vector<vector<double>> regrets;

    for (int i = 0; i < numRuns; i++) {
        fprintf(stderr, "run: %d\n", i);
        LinUCB agent = LinUCB(10, 5, 1, 1, 0);
        RandomEnv env = RandomEnv(10, 5, seed + i);

        pair<vector<double>, vector<double>> runResults = runAgent(agent, env, numEpisodes, episodeLength);
        returns.push_back(runResults.first);
        regrets.push_back(runResults.second);
    }

    // write to file
    writeResults(returns, regrets);
}

    /*
    // break into 2 pieces
    torch::Tensor ar1 = torch::arange(0,4);
    torch::Tensor m1 = torch::index_select(torch::index_select(M, 0, ar1), 1, ar1);
    torch::Tensor ar2 = torch::arange(0,4);
    torch::Tensor m2 = torch::index_select(torch::index_select(M, 0, ar1), 1, ar1);
    // calculate inverse
    torch::Tensor m1_inv = torch::inverse(m1);
    torch::Tensor m2_inv = torch::inverse(m2);
    // stitch back together
    torch::Tensor Mm_inv = torch::block_diag({m1_inv, m2_inv});
    */
/*
int partitions = 4;
vector<torch::Tensor> ar(partitions);

torch::Tensor calculate_inverse(torch::Tensor t, int partitions) {
    vector<torch::Tensor> tl(partitions); // list of tensors to rebuild

    for (int i = 0; i < partitions; i++) {
        tl[i] = torch::inverse(torch::index_select(torch::index_select(t, 0, ar[i]), 1, ar[i]));
    }

    return torch::block_diag(tl);
}
*/

int main(int argc, char *argv[]) {
    /*
    int num_iter = atoi(argv[1]);
    bool split = false;
    if (argc > 2 && strcmp(argv[2], "-s") == 0) {
        split = true;
    }

    int state_size = 25;

    for (int i = 0; i < partitions; i++) {
        ar[i] = torch::arange(i * state_size, (i+1) * state_size);
    }

    // create random tensor
    torch::Tensor M = torch::block_diag({torch::randn({state_size, state_size}),
            torch::randn({state_size, state_size}), torch::randn({state_size, state_size}),
            torch::randn({state_size, state_size})});

    if (split) {
        printf("split iterations: %d\n", num_iter);
        for (int i = 0; i < num_iter; i++) {
            torch::Tensor M_inv = calculate_inverse(M, partitions);
        }
    } else {
        printf("no split iterations: %d\n", num_iter);
        for (int i = 0; i < num_iter; i++) {
            torch::Tensor M_inv = torch::inverse(M);
        }
    }
    // compare against inverse of original
    //torch::Tensor M_inv = torch::inverse(M);

    //cout << M << endl;
    //cout << M_inv << endl;
    return 0;
    */

    /*
    torch::Tensor i = torch::eye(5);
    torch::Tensor v = torch::ones(5);
    v[4] = 5;

    cout << i << endl;
    cout << v << endl;
    v = v* i;
    cout << v << endl;

    return 0;
    */

    int numRuns = 5; //30;
    int numEpisodes = 5000;
    int episodeLength = 10;
    run_test(numRuns, numEpisodes, episodeLength, 382.4268);

    return 0;
}
