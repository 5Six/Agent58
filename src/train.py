import gym, torch, torch.optim
import numpy as np, time
import matplotlib.pyplot as plt
from collections import deque

from net import Q_net, Dueling_DQN
from replay import ReplayBuffer


# hyperparameters
N_episodes = 3000  # 10000
env_version = 5  # cartpole version 0 or 1

method = 'double'  # method for evaluating the targets; double stands for DDQN
# method = 'vanilla'
# method = 'dueling'

learning_rate = 1e-4
Size_replay_buffer = 50000  # in time steps
epsilon_start = 1  # epsilon for epsilon greedy algorithm
epsilon_end = 0.02
eps_anneal_period = 10000  # simple linear annealing
Size_minibatch = 32
net_update_period = 500  # after how many minibatches should the target computing net be updated
gamma = 1
l2_regularization = 0  # L2 regularization coefficient
plot_freq = 10

custom_name = 'double2'
net_save_path = f'net/net_boxing-v{env_version}_{method}DQN_{custom_name}.pth'
plot_save_path = f'plot/plot_boxing-v{env_version}_{method}DQN_{custom_name}.png'
device = "cuda"

if env_version == 1:
    Pass_score = 499  # usually 475
elif env_version == 0:
    Pass_score = 500  # was 199
elif env_version == 5:
    Pass_score = 5000  # was 199, set to ridiculously high number for all episodes to run
else:
    assert False, "wrong env_version, should be 0 or 1 or 5(integer)"

if method == 'dueling':
    net = Dueling_DQN()  # net that determines policy to follow (except when randomly choosing actions during training)
    net.to(device)
    target_net = Dueling_DQN()  # net that computes the targets
    target_net.to(device)
    target_net.load_state_dict(net.state_dict())  # copying all network parameters
    target_net.eval()
else: # vanilla or double
    net = Q_net()  # net that determines policy to follow (except when randomly choosing actions during training)
    net.to(device)
    target_net = Q_net()  # net that computes the targets
    target_net.to(device)
    target_net.load_state_dict(net.state_dict())  # copying all network parameters
    target_net.eval()

# loss_function = torch.nn.MSELoss()
loss_function = torch.nn.SmoothL1Loss()  # Huber loss
# optimizer = torch.optim.RMSprop(net.parameters(), weight_decay=l2_regularization)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_regularization)

replay_buffer = ReplayBuffer(Size_replay_buffer, keys=['state', 'action', 'reward', 'state_next', 'done'])

env = gym.make('ALE/Boxing-ram-v'+str(env_version))
epsilon = epsilon_start 

backprops_total = 0  # to track when to update the target net
running_loss = 0
# score is a number of steps made within one episode
avg_score_best = 0  # network will be saved only if a mean of last 100 episodes exceeds previous best mean of 100
latest_scores = deque(maxlen=100)
avg_score_history = []

state_curr = env.reset()
state_prev = state_curr
score = 0  # score per episode
t0 = time.time()
for ep in range(N_episodes): # ep stands for episode
    done = False
    while not done:
    # for step in range(T_max+2):
        # choose an action:
        state = torch.from_numpy(np.concatenate((state_curr, state_curr-state_prev))).float()
        state = state.to(device)  # input to network
        if np.random.rand() < epsilon:
            action = np.random.randint(18)
        else:
            net.eval()
            q = net(state.view(1, -1))
            action = np.argmax(q.detach().cpu().numpy())

        s_next, reward, done, _ = env.step(action)

        score += reward
        # store the experience
        state_next = torch.from_numpy(np.concatenate((s_next, s_next-state_curr))).float()
        replay_buffer.append((state, action, reward, state_next, done))

        if done:
            latest_scores.append(score)
            avg_score_history.append(np.mean(latest_scores))
            score = 0
            state_curr = env.reset()
            state_prev = state_curr
        else:
            state_prev = state_curr
            state_curr = s_next

        if epsilon > epsilon_end:  # annealing
            epsilon -= (epsilon_start-epsilon_end)/eps_anneal_period

        # train on one minibatch:
        if len(replay_buffer) < Size_minibatch:
            continue
        net.train()
        minibatch_ids = np.random.choice(len(replay_buffer), Size_minibatch)
        minibatch = replay_buffer.get(minibatch_ids)
        xs, actions, rs, next_xs, dones = minibatch.values()
        xs = torch.stack(xs).to(device)  # list of tensors -> tensor
        next_xs = torch.stack(next_xs).to(device)
        rs = np.array(rs)
        final_state_ids = np.nonzero(dones)  # will be needed to calculate targets for terminal states properly
        rs = torch.from_numpy(rs).float()

        if method == 'double':
            # finding targets by double DQN method
            with torch.no_grad():
                net.eval()
                Q_next = net(next_xs)
                Q_next_ = target_net(next_xs)
            net.train()
            optimizer.zero_grad()
            Q = net(xs)
            Q_next_max, Q_next_argmax = torch.max(Q_next, 1)
            V_next = torch.gather(Q_next_, 1, Q_next_argmax.view(-1, 1)).squeeze()

        else: # vanilla
            # finding targets by vanilla method
            with torch.no_grad():
                # print(next_xs)
                Q_next_ = target_net(next_xs)
            optimizer.zero_grad()
            Q = net(xs)
            Q_next_max, Q_next_argmax = torch.max(Q_next_, 1)
            V_next = Q_next_max

        V_next[final_state_ids] = 0  # terminal states should have V(s) = max(Q(s,a)) = 0
        Q_target = (rs.to(device) + gamma*V_next).to(device)
        # backprop only on actions that actually occured at corresponding states
        actions = torch.tensor(actions).view(-1, 1)
        Q_relevant = torch.gather(Q, 1, actions.to(device)).squeeze()
        loss = loss_function(Q_relevant, Q_target)
        loss.backward()
        optimizer.step()
        running_loss = loss.item() if running_loss == 0 else 0.99*running_loss + 0.01*loss.item()

        backprops_total += 1
        if backprops_total % net_update_period == 0:
            # if running_score > running_score_best:  ## useless
            target_net.load_state_dict(net.state_dict())
            # else:  ## useless
            #     net.load_state_dict(target_net.state_dict())  ## useless
        ep_played = ep + 1
        if done and ep_played % plot_freq == 0:
            print("ep: {}, buf_len: {}, epsilon: {:.3f}, time: {:.2f}s, running_loss: {:.3f}, last 100 avg score: {:.1f}".
                  format(ep_played, len(replay_buffer), epsilon, time.time()-t0,
                                                                           running_loss, np.mean(latest_scores)))
        if done and ep_played % 10 == 0 and np.mean(latest_scores) > avg_score_best:
            torch.save(net.state_dict(), net_save_path) # saves policy net weights, try out target net
            avg_score_best = np.mean(latest_scores)
            if avg_score_best > Pass_score:
                print('latest 100 average score: {}, pass score: {}, test is passed'.format(avg_score_best, Pass_score))
                plt.close('all')
                plt.plot(avg_score_history)
                plt.xlabel('episodes')
                plt.ylabel('last 100 average score')
                plt.savefig(plot_save_path)
                exit(0)
            # print("net saved to '{}'".format(net_save_path))

        if done and ep_played % plot_freq == 0: # done flag not done correctly?
            plt.close('all')
            plt.plot(avg_score_history)
            plt.xlabel('episodes')
            plt.ylabel(f'last {plot_freq} average score')
            plt.savefig(plot_save_path)
        if done:
            break
env.close()
