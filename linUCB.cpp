#include "linUCB.h"
#include <iostream>

using namespace std;

LinUCB::LinUCB(int observeDims, int numActions, double regularizer, double delta, int seed) {
    this->observeDims = observeDims;
    this->numActions = numActions;
    this->regularizer = regularizer;
    this->delta = delta;

    this->theta = torch::rand(this->observeDims * this->numActions);
    this->V = torch::eye(this->observeDims * this->numActions) * this->regularizer;
    this->b = torch::zeros(this->observeDims * this->numActions);

    this->timestep = 1;

    generator.seed(seed);
}


#ifdef SEPARATE_UPDATES
void LinUCB::updateAgent(torch::Tensor observation, int action, double reward) {
    this->timestep++;
    torch::Tensor prevActionContext = createActionContext(observation, action);
    updateTheta(prevActionContext, reward);
}

#else

int LinUCB::start(torch::Tensor observation) {
    int action = selectAction(observation);
    this->oldActionContext = createActionContext(observation, action);

    return action;
}

int LinUCB::step(double reward, torch::Tensor observation) {
    this->timestep++;

    updateTheta(this->oldActionContext, reward);

    int action = selectAction(observation);
    this->oldActionContext = createActionContext(observation, action);

    return action;
}

void LinUCB::end(double reward) {
    this->timestep++;

    updateTheta(this->oldActionContext, reward);
}
#endif /* SEPARATE_UPDATES */

int LinUCB::selectAction(torch::Tensor observation, vector<int> available_actions) {
    double beta = sqrt(this->regularizer) + sqrt(2 * log(1 / this->delta) + this->numActions *
            log(1 + (this->timestep-1) / (this->regularizer * this->numActions)));

    torch::Tensor ucb = torch::zeros(this->numActions);
    torch::Tensor inverseV = torch::inverse(this->V);

    for (int action : available_actions) {
        torch::Tensor actionContext = createActionContext(observation, action);
        ucb[action] = actionContext.dot(this->theta) + beta *
            torch::sqrt(torch::matmul(actionContext, inverseV).dot(actionContext));
    }

    // select highest ucb value
    vector<int> actions = {available_actions[0]};
    double actionValue = ucb[actions[0]].item().to<double>();
    for (int i = 1; i < available_actions.size(); i++) {
        int action = available_actions[i];
        double tmpActionValue = ucb[action].item().to<double>();

        if (tmpActionValue > actionValue) {
            actions = {action};
            actionValue = tmpActionValue;
        } else if (tmpActionValue == actionValue) {
            actions.push_back(action);
        }
    }

    return actions[generator() % actions.size()];
}

torch::Tensor LinUCB::createActionContext(torch::Tensor observation, int action) {
    torch::Tensor actionContext = torch::empty(0);

    for (int i = 0; i < this->numActions; i++) {
        if (i == action) {
            actionContext = torch::cat({actionContext, observation}, 0);
        } else {
            actionContext = torch::cat({actionContext, torch::zeros(this->observeDims)}, 0);
        }
    }

    return actionContext;
}

void LinUCB::updateTheta(torch::Tensor oldActionContext, double reward) {
    this->V = this->V + torch::outer(oldActionContext, oldActionContext);

    this->b = this->b + reward * oldActionContext;

    this->theta = torch::matmul(torch::inverse(this->V), this->b);
}
