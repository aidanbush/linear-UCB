

class LinUCB():
    # c context dimensions
    # a number of actions
    # theta a x c matrix of weights

    V = None

    def

    def start(context):
        pass

    def step(context, reward):
        # update old action
        # V = V + old_action dot transpose(old_action)
        # b = b + old_context dot old_action
        # theta = V^inverse * b

        # select new action
        # context is a vertical vector of size c
        # get action set - do nothing
        # beta = root(lambda) + root(2 log(1 / delta) + d log(1 + (t - 1) / lambda * d))
        # compute UCB
            # ucb = transpose(context) dot theta + beta root(transpose(context) dot V^inverse dot context)
        # action = argmax(ucb)
        # return action
        pass

    def end(reward):
        pass

