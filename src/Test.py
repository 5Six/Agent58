from Net import Net
import torch, gym
import numpy as np

def test():
	episodes = 1
	# method = "vanilla"
	method = "double"
	custom_name = "test_69"
	net_path = f"net/net_boxing-v5_{method}DQN_{custom_name}.pth"
	net = Net(128, 18)
	net.load_state_dict(torch.load(net_path))
	net.eval()

	env = gym.make("ALE/Boxing-ram-v5", render_mode='human')
	scores = []

	for episode in range(episodes):
		state_prev = state_curr = env.reset()
		score = 0

		while True:
			x = torch.from_numpy(np.concatenate((state_curr, state_curr - state_prev))).float()
			q = net(x)
			qmax, a = torch.max(q, 0)
			a = a.item()
			state_prev = state_curr
			state_curr, reward, done, _ = env.step(a)
			score += reward

			if done:
				break

		scores.append(score)
		print("ep {}, score: {}".format(episode, score))

	env.close()
	print("average score: {}, minimal score: {}".format(np.mean(scores), min(scores)))


if __name__ == "__main__":
    test()