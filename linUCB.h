#ifndef LINUCB_H
#define LINUCB_H

#include <torch/torch.h>

using namespace std;

class LinUCB {
    public:
        LinUCB(int observeDims, int numActions, double regularizer, double delta);

        int start(torch::Tensor observation);
        int step(double reward, torch::Tensor observation);
        void end(double reward);

    private:

        int observeDims;
        int numActions;
        double regularizer;
        double delta;

        torch::Tensor V;
        torch::Tensor theta;
        torch::Tensor b;

        int timestep;

        torch::Tensor oldActionContext;

        // helpers
        int selectAction(torch::Tensor observation);
        torch::Tensor createActionContext(torch::Tensor observation, int action);

        void updateTheta(double reward);
};

#endif /* LINUCB_H */
