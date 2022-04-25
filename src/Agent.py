from collections import namedtuple
import copy
from typing import Union
import torch
from torch import nn
from torch import optim
import numpy as np
from Replay import ReplayMemory
from Net import Net


class Agent:
    def __init__(
        self,
        config: dict,
        state_space: int,
        action_space: int,
        device: str,
    ) -> None:
        """
        __init__ _summary_

        Args:
            learning_rate (float): _description_
            state_space (int): _description_
            action_space (int): _description_
            gradient_algo (str): _description_
            device (_type_): _description_
            seed (int, optional): _description_. Defaults to 10.
            memory_capacity (int, optional): _description_. Defaults to 1000.
            batch_size (int, optional): _description_. Defaults to 32.
        """

        self.learning_rate = config['alpha']
        self.seed = config['seed']
        torch.manual_seed(self.seed)
        self.memory_capacity = config['memory_capacity']
        self.batch_size = config['batch_size']
        self.action_space = action_space
        self.state_space = state_space
        self.gradient_algo = config['gradient_algorithm']
        self.device = device
        self.buffer_tuple = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "terminal")
        )
        self.buffer = ReplayMemory(self.buffer_tuple, self.memory_capacity)
        self.action_value_network = Net(state_space, action_space).to(device)
        self.target_value_network = copy.deepcopy(self.action_value_network)
        self.optimiser = self.get_optimisation()

    def choose_action(self, epsilon, state):
        """
        choose_action _summary_

        Args:
            epsilon (_type_): _description_
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        if np.random.random() < epsilon:
            action = torch.tensor(
                [[np.random.randint(self.action_space)]],
                device=self.device,
                dtype=torch.long,
            )
        else:
            with torch.no_grad():
                #ERRORING - state input on cpu but nnexit() on cuda
                # print(self.action_value_network(state))
                action = torch.argmax(self.action_value_network(state)).view(1,1)

        return action

    def store_transition(self, transition: tuple) -> None:
        """
        store_transition _summary_

        Args:
            transition (tuple): _description_
        """
        self.buffer.push(transition[0], transition[1], transition[2], transition[3], transition[4])

    @property
    def sample_experience(self):
        """
        sample_experience _summary_

        Returns:
            _type_: _description_
        """
        if len(self.buffer) < self.batch_size:
            return None
        return self.buffer.sample(self.batch_size)

    def learn(self, gamma, experience) -> tuple:
        """
        learn _summary_

        Args:
            gamma (_type_): _description_
            experience (_type_): _description_

        Returns:
            tuple: _description_
        """

        batch = self.buffer_tuple(*zip(*experience))
    
        states = torch.cat(batch.state)
        test = states.shape
        actions = torch.cat(batch.action)
        next_states = torch.cat(batch.next_state)
        rewards = torch.cat(batch.reward)
        terminals = torch.cat(batch.terminal)

        with torch.no_grad():
            target_q_values = self.target_value_network(next_states)
        self.optimiser.zero_grad()

        current_q_values = self.action_value_network(states)
        max_next_q_values = torch.max(target_q_values, 1)[0]
               
        # terminal states should have V(s) = max(Q(s,a)) = 0
        max_next_q_values[terminals] = 0

        expected_q_values = (rewards + (gamma * max_next_q_values).squeeze())

        relavent_q_values = torch.gather(current_q_values, 1, actions.view(-1, 1)).squeeze()

        return relavent_q_values, expected_q_values, expected_q_values

    def get_loss(self, current, expected, function) -> Union[nn.HuberLoss, nn.MSELoss]:
        #print(current,"                ", expected)
        if function.lower() == "huberloss":
            loss_function = nn.SmoothL1Loss()
        else:
            loss_function = nn.MSELoss()

        return loss_function(current, expected)

    def gradient_decent(self, loss: Union[nn.HuberLoss, nn.MSELoss]) -> None:
        loss.backward()
        self.optimiser.step()

    def update_target_network(self) -> None:
        self.target_value_network.load_state_dict(self.action_value_network.state_dict())

    def get_optimisation(self):

        if self.gradient_algo.lower() == "rmsprop":
            optimiser = optim.RMSprop(self.action_value_network.parameters(), lr=self.learning_rate)
        else:
            optimiser = optim.Adam(self.action_value_network.parameters(), lr=self.learning_rate)

        return optimiser

    def get_weights(self, net_save_path):
        torch.save(self.action_value_network.state_dict(), net_save_path)