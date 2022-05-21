from Net import Net
from Net import Dueling_DQN
import torch, gym
import numpy as np
from Utils import nparray_to_tensor

def test():
	episodes = 1000
	# method = "vanilla"
	method = "double"
	custom_name = "best_1"
	#using_per = ""
	using_dueling = True
    
	#net_path = f"net/net_boxing-v5_{method}DQN_{custom_name}{using_per}_action_net.pth"
	net_path = "net/Dueling Double DQN with PER.pth"
	path_to_file = "Results/PER.txt"
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	
	if (using_dueling):
		net = Dueling_DQN(256, 18)
	else:
		net = Net(128,18)
		
	net.load_state_dict(torch.load(net_path))
	net.eval()
	net.to(device=device)

	env = gym.make("ALE/Boxing-ram-v5", render_mode = "human")
	scores = []

	
	f = open(path_to_file, 'w')

	for episode in range(episodes):
		score = 0
		state_current = env.reset()
		state_current = nparray_to_tensor(state_current, device)
		state_previous = state_current

		while True:		
			state_with_diff = torch.cat((state_current[0], state_current[0] - state_previous[0]))
			q = net(state_with_diff)
			qmax, a = torch.max(q, 0)
			action = a.item()

			next_state, reward, done, _ = env.step(action)
			next_state = nparray_to_tensor(next_state, device)
			next_state_with_diff = torch.cat((next_state[0], next_state[0] - state_current[0]))

			state_previous = state_current
			state_current = next_state

			score += reward

			if done:
				break

		scores.append(score)
		f.write(str(score) + "\n")
		print("ep {}, score: {}".format(episode, score))

	env.close()
	print("average score: {}, minimal score: {}".format(np.mean(scores), min(scores)))


if __name__ == "__main__":
    test()
