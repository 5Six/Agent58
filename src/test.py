from net import Q_net
import torch, gym
import numpy as np

# render = True
custom_name = '10k_eps'
N_ep = 1
env_version = 5
# method = 'double'
method = 'vanilla'
net_load_path = f'net/net_boxing-v{env_version}_{method}DQN_{custom_name}.pth'
net = Q_net()
net.load_state_dict(torch.load(net_load_path))
net.eval()

env = gym.make('ALE/Boxing-ram-v'+str(env_version), render_mode='human')
scores = []
for ep in range(N_ep):
    s_cur = env.reset()
    s_prev = s_cur
    score = 0
    while True:
        # if render:
        #     env.render()
        x = torch.from_numpy(np.concatenate((s_cur, s_cur - s_prev))).float()
        q = net(x.view(1,-1)).squeeze()
        qmax, a = torch.max(q, 0)
        a = a.item()
        s_prev = s_cur
        s_cur, r, done, _ = env.step(a)
        score += r
        if done:
            break
    scores.append(score)
    print("ep {}, score: {}".format(ep, score))
env.close()
print("average score: {}, minimal score: {}".format(np.mean(scores), min(scores)))

