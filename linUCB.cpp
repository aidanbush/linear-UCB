#include "linUCB.h"
#include <iostream>

using namespace std;

LinUCB::LinUCB(int observeDims, int numActions, double regularizer, double delta) {
    this->observeDims = observeDims;
    this->numActions = numActions;
    this->regularizer = regularizer;
    this->delta = delta;

    this->theta = torch::rand(this->observeDims * this->numActions);
    this->V = torch::eye(this->observeDims * this->numActions) * this->regularizer;
    this->b = torch::zeros(this->observeDims * this->numActions);

    this->timestep = 1;
}

int LinUCB::start(torch::Tensor observation) {
    return selectAction(observation);
}

int LinUCB::step(double reward, torch::Tensor observation) {
    this->timestep++;

    updateTheta(reward);

    return selectAction(observation);
}

void LinUCB::end(double reward) {
    this->timestep++;

    updateTheta(reward);
}

int LinUCB::selectAction(torch::Tensor observation) {
    double beta = sqrt(this->regularizer) + sqrt(2 * log(1 / this->delta) + this->numActions *
            log(1 + (this->timestep-1) / (this->regularizer * this->numActions)));

    torch::Tensor ucb = torch::zeros(this->numActions);
    torch::Tensor inverseV = torch::inverse(this->V);

    for (int action = 0; action < this->numActions; action++) {
        torch::Tensor actionContext = createActionContext(observation, action);
        ucb[action] = actionContext.dot(this->theta) + beta *
            torch::sqrt(torch::matmul(actionContext, inverseV).dot(actionContext));
    }

    // select highest ucb value
    int action = 0;
    double actionValue = ucb[0].item().to<double>();
    for (int i = 1; i < this->numActions; i++) {
        double tmpActionValue = ucb[i].item().to<double>();
        if (tmpActionValue > actionValue) {
            action = i;
            actionValue = tmpActionValue;
        //} else if (ucb[i] == actionValue) {
        }
    }

    this->oldActionContext = createActionContext(observation, action);

    return action;
}

torch::Tensor LinUCB::createActionContext(torch::Tensor observation, int action) {
    // auto options = torch::TensorOptions().device(torch::kCPU);
    // torch::Tensor actionContext = torch::Tensor({}, 0, options);
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

void LinUCB::updateTheta(double reward) {
    this->V = this->V + torch::outer(this->oldActionContext, this->oldActionContext);

    this->b = this->b + reward * this->oldActionContext;

    this->theta = torch::matmul(torch::inverse(this->V), this->b);
}
