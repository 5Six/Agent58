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
        learning_rate: float,
        state_space: int,
        action_space: int,
        gradient_algo: str,
        device,
        seed: int = 10,
        memory_capacity: int = 1000,
        batch_size: int = 32,
    ) -> None:

        self.learning_rate = learning_rate
        self.seed = seed
        torch.manual_seed(self.seed)
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.action_space = action_space
        self.state_space = state_space
        self.gradient_algo = gradient_algo
        self.device = device
        self.buffer_tuple = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "terminal")
        )
        self.buffer = ReplayMemory(self.buffer_tuple, self.memory_capacity)
        self.action_value_network = Net(state_space, action_space).to(device)
        self.target_value_network = copy.deepcopy(self.action_value_network)
        self.optimiser = self.get_optimisation()

    def choose_action(self, epsilon, state):

        # add epsilon decay here
        # make device parameters a valiable

        if np.random.random() < epsilon:
            action = torch.tensor(
                [[np.random.choice(self.action_space)]],
                device=self.device,
                dtype=torch.long,
            )
        else:
            with torch.no_grad():

                # ERRORING - state input on cpu but nn on cuda
                # action = self.action_value_network(state).max(1)[1].view(1, 1)

                action = torch.tensor(
                    [[np.random.choice(self.action_space)]],
                    device=self.device,
                    dtype=torch.long,
                )

        return action

    def store_transition(self, transition: tuple) -> None:
        self.buffer.push(transition[0], transition[1], transition[2], transition[3], transition[4])

    def sample_experience(self):
        if len(self.buffer) < self.batch_size:
            return None
        return self.buffer.sample(self.batch_size)

    def learn(self, gamma, experience) -> tuple:

        # CHECK
        batch = self.buffer_tuple(*zip(*experience))

        # needed??
        # non_final_mask = torch.tensor(
        #     tuple(map(lambda s: s is not True, batch.terminal)),
        #     device=self.device,
        #     dtype=torch.bool,
        # )

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        next_states = torch.cat(batch.next_state)
        rewards = torch.cat(batch.reward)
        terminals = torch.cat(batch.terminal)

        # non terminal next states
        # non_final_next_states = torch.masked_select(next_states, terminals)
        non_terminal_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # current Q values are estimated by the network for all actions
        current_q_value = self.action_value_network(states).gather(1, actions)

        # expected Q values are estimated from actions which gives maximun Q value
        max_next_q_value = torch.zeros(self.batch_size, device=self.device)

        max_next_q_value[terminals] = (
            self.action_value_network(non_final_next_states).max(1)[0].detach()
        )
        expected_q_value = rewards + (gamma * max_next_q_value)

        return current_q_value, expected_q_value

    def get_loss(self, current, expected, function) -> Union[nn.HuberLoss, nn.MSELoss]:

        # CHECK
        if function.lower() == "huberloss":
            loss = nn.HuberLoss(current, expected.unsqueeze(1))
        else:
            loss = nn.MSELoss(current, expected.unsqueeze(1))

        return loss

    def gradient_decent(self, loss):

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass

    def update_target_network(self):
        pass

    def get_optimisation(self):

        if self.gradient_algo.lower() == "rmsprop":
            optimiser = optim.RMSprop(self.action_value_network.parameters(), lr=self.learning_rate)
        else:
            optimiser = optim.Adam(self.action_value_network.parameters(), lr=self.learning_rate)

        return optimiser
