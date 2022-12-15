#ifndef LINUCB_H
#define LINUCB_H

#define SEPARATE_UPDATES

#include <torch/torch.h>
#include <random>

using namespace std;

class LinUCB {
    public:
        LinUCB(int observeDims, int numActions, double regularizer, double delta, int seed);

#ifdef SEPARATE_UPDATES
        void updateAgent(torch::Tensor observation, int action, double reward);
        int selectAction(torch::Tensor observation, vector<int> available_actions);
#else /* SEPARATE_UPDATES */
        int start(torch::Tensor observation);
        int step(double reward, torch::Tensor observation);
        void end(double reward);
#endif /* SEPARATE_UPDATES */

    private:

        int observeDims;
        int numActions;
        double regularizer;
        double delta;

        torch::Tensor V;
        torch::Tensor theta;
        torch::Tensor b;

        int timestep;

#ifndef SEPARATE_UPDATES
        torch::Tensor oldActionContext;
#endif /* SEPARATE_UPDATES */

        // helpers
#ifndef SEPARATE_UPDATES
        int selectAction(torch::Tensor observation);
#endif /* SEPARATE_UPDATES */
        torch::Tensor createActionContext(torch::Tensor observation, int action);

        void updateTheta(torch::Tensor oldActionContext, double reward);

        default_random_engine generator;
};

#endif /* LINUCB_H */
