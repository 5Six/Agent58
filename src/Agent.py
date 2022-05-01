from collections import namedtuple
import os
import copy
from typing import Union
import torch
from torch import nn
from torch import optim
import numpy as np
import torch.nn.functional as F
from Replay import PriorityReplayMemory
from Replay import ReplayMemory
from Net import Net, Dueling_DQN


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
        self.config = config
        self.learning_rate = config['alpha']
        self.seed = config['seed']
        torch.manual_seed(self.seed)
        self.memory_capacity = config['memory_capacity']
        self.batch_size = config['batch_size']
        self.action_space = action_space
        self.state_space = state_space
        self.gradient_algo = config['gradient_algorithm']
        self.device = device
        self.LOSS_FUNCTION = config['loss_function']
        self.buffer_tuple = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "terminal")
        )

        if config['per'] == "True":
            self.buffer = PriorityReplayMemory(self.buffer_tuple, self.memory_capacity, config['per_offset'], config['per_alpha'], config['per_beta'], config['per_beta_increment_per_sampling'])
        else:
            self.buffer = ReplayMemory(self.buffer_tuple, self.memory_capacity)
        if config['dueling'] == "True":
            self.action_value_network = Dueling_DQN(state_space, action_space).to(device)
            self.target_value_network = copy.deepcopy(self.action_value_network)
        else:
            self.action_value_network = Net(state_space, action_space).to(device)
            self.target_value_network = copy.deepcopy(self.action_value_network)
        self.optimiser = self.get_optimisation()
        self.method = config['method']
        self.net_file_name = self.get_file_name(config)
        
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

    def store_transition(self, transition: tuple, gamma) -> None:
        if self.config['per'] == "True":
            reward = transition[3].item()
            action = transition[1].item()
            q_value = self.action_value_network(transition[0]).squeeze()[action].item()
            max_q_value_next = torch.max(self.target_value_network(transition[2])[0]).item()
            if (transition[4] == True):
                target_q_value = reward
            else:
                target_q_value = reward + gamma * max_q_value_next
            E = 0.5
            td_error = abs(q_value - target_q_value)
            self.buffer.push(td_error, (transition[0], transition[1], transition[2], transition[3], transition[4]))
        else:
            self.buffer.push(transition[0], transition[1], transition[2], transition[3], transition[4])

    @property
    def sample_experience(self):
        """
        sample_experience _summary_

        Returns:
            _type_: _description_
        """
        if self.config['per'] != "True":
            if len(self.buffer) < self.batch_size:
                return None

        return self.buffer.sample(self.batch_size)
        

    def learn(self, gamma, experience, idxs, weights) -> tuple:
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
        actions = torch.cat(batch.action)
        next_states = torch.cat(batch.next_state)
        rewards = torch.cat(batch.reward)
        terminals = torch.cat(batch.terminal)

        if self.method == "double":
            with torch.no_grad():
                self.action_value_network.eval()
                action_q_values = self.action_value_network(next_states)
                target_q_values = self.target_value_network(next_states)
            self.action_value_network.train()
            self.optimiser.zero_grad()
            current_q_values = self.action_value_network(states)
            argmax_next_q_values = torch.max(action_q_values, 1)[1]
            max_next_q_values = torch.gather(target_q_values, 1, argmax_next_q_values.view(-1, 1)).squeeze()
        else:
            with torch.no_grad():
                target_q_values = self.target_value_network(next_states)
            self.optimiser.zero_grad()
            current_q_values = self.action_value_network(states)
            max_next_q_values = torch.max(target_q_values, 1)[0]
               
        # terminal states should have V(s) = max(Q(s,a)) = 0
        max_next_q_values[terminals] = 0

        expected_q_values = (rewards + (gamma * max_next_q_values).squeeze())

        relavent_q_values = torch.gather(current_q_values, 1, actions.view(-1, 1)).squeeze()

        if self.config['per'] == "True":
            pred = relavent_q_values

            target = expected_q_values

            errors = torch.abs(pred - target)

            for i in range(self.batch_size):
                idx = idxs[i]       
                self.buffer.update(idx, errors[i].item())
            
            weights =  torch.from_numpy(weights).to(self.device)
            loss = self.get_loss(relavent_q_values, expected_q_values, self.LOSS_FUNCTION)
            loss = (weights * loss).mean()
            loss.backward()
            #and train
            self.optimiser.step()
        else:
            
            loss = self.get_loss(relavent_q_values, expected_q_values, self.LOSS_FUNCTION)
            self.gradient_decent(loss)
             
        return relavent_q_values, expected_q_values, expected_q_values

    def get_loss(self, current, expected, function) -> Union[nn.HuberLoss, nn.MSELoss]:
        if function.lower() == "huberloss":
            loss_function = nn.SmoothL1Loss()
        else:
            loss_function = nn.MSELoss()

        return loss_function(current, expected).to(self.device)

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

    def save_weights(self):
        torch.save(
            self.action_value_network.state_dict(),
            f"net/{self.net_file_name}_action.pth"
            )
        torch.save(
            self.target_value_network.state_dict(),
            f"net/{self.net_file_name}_target.pth"
            )

    def get_file_name(self, config):
        custom_name = ""
        if config["per"] == "True":
            custom_name += "_PER"
        if config["dueling"] == "True":
            custom_name += "_Dueling"
        if config["custom_name"]:
            custom_name += f"_{config['custom_name']}"

        file_name = f"net_boxing-v5_{self.method}DQN{custom_name}"

        i = 1
        while os.path.exists(f"net/{file_name}_{i}_action.pth"):
            i += 1
    
        return f"{file_name}_{i}"
