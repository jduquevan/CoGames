import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from functools import reduce

from .coin_game import CoinGame
from .models import MLPModel, ConvModel, ReplayMemory, Transition, A2CTransition, \
                    Actor, NashACTransition, RfNashACTransition, LSTMModel, LinearQHead

def construct_model(model_type,
                    in_size,
                    out_size,
                    hidden_size,
                    num_layers,
                    batch_norm=False,
                    in_channels=0,
                    h=0,
                    w=0,
                    lstm_out=20):
    if model_type == "mlp":
        model = MLPModel(in_size=in_size,
                         out_size=out_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_norm=batch_norm)
    elif model_type == "conv":
        model = ConvModel(in_channels=history_len,
                          h=h,
                          w=w,
                          out_size=out_size)
    elif model_type == "lstm":
        model = LSTMModel(in_size=in_size,
                          out_size=out_size,
                          hidden_size=hidden_size,
                          lstm_out=lstm_out,
                          num_layers=num_layers)
    else:
        raise Exception(model_type, " is not a supported model type for Agents")
    return model

#TODO: Refactor so that only essentials are stored in BaseAgent, update config appropiately
class BaseAgent():
    def __init__(self,
                 device,
                 gamma,
                 n_actions,
                 hidden_size,
                 num_layers,
                 batch_norm, 
                 obs_shape,
                 buffer_size,
                 use_actions, 
                 model_type="mlp",
                 opt_type="rmsprop",
                 batch_size=128,
                 history_len=0,
                 temperature=1,
                 is_pc=False,
                 is_nash=False,
                 is_rf_nash=False,
                 lstm_out=20):
        self.steps_done = 0
        self.device = device
        self.gamma = gamma
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.lstm_out = lstm_out
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.obs_shape = obs_shape
        self.batch_size = batch_size
        self.history_len = history_len
        self.temperature = temperature
        self.buffer_size = buffer_size
        self.model_type = model_type.lower()
        self.opt_type = opt_type.lower()
        self.is_pc = is_pc
        self.use_actions = use_actions
        self.obs_size = reduce(lambda a, b: a * b, self.obs_shape)

        self.obs_history = []
        self.act_history = []

        if self.is_pc:
            self.act_obs_size = self.obs_size + 2 * self.n_actions
        elif self.use_actions:
            self.act_obs_size = self.obs_size + 2
        else:
            self.act_obs_size = self.obs_size

        if self.model_type == "lstm":
            self.val_obs_size = self.act_obs_size
        else:
            self.val_obs_size = self.history_len * self.act_obs_size

        if self.model_type == "mlp":
            self.q_net = MLPModel(in_size=self.obs_size,
                                  out_size=self.n_actions,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  batch_norm=self.batch_norm)
            self.t_net = MLPModel(in_size=self.obs_size,
                                  out_size=self.n_actions,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  batch_norm=self.batch_norm)
        elif self.model_type == "conv":
            self.q_net = ConvModel(in_channels=history_len,
                                  h=self.obs_shape[0],
                                  w=self.obs_shape[1],
                                  out_size=self.n_actions)
            self.t_net = ConvModel(in_channels=history_len,
                                  h=self.obs_shape[0],
                                  w=self.obs_shape[1],
                                  out_size=self.n_actions)
        # elif self.model_type == "lstm":
        #     self.obs_size = reduce(lambda a, b: a * b, self.obs_shape) + 2
        #     if is_nash:
        #         self.val_obs_size = self.obs_size + 2 * self.n_actions
        #     self.q_net = LSTMModel(in_size=self.val_obs_size,
        #                            out_size=self.val_out_size,
        #                            hidden_size=self.hidden_size,
        #                            lstm_out=self.lstm_out,
        #                            num_layers=self.num_layers)
        #     self.t_net = LSTMModel(in_size=self.val_obs_size,
        #                            out_size=self.val_out_size,
        #                            hidden_size=self.hidden_size,
        #                            lstm_out=self.lstm_out,
        #                            num_layers=self.num_layers)
        # else:
        #     raise Exception(self.model_type, " is not a supported model type for Agents")

        # if opt_type.lower() == "rmsprop":
        #     self.optimizer = optim.RMSprop(self.q_net.parameters())
        
        # self.t_net.load_state_dict(self.q_net.state_dict())
        # self.q_net.to(self.device)
        # self.t_net.to(self.device)

    def update_history(self, act, state):
        self.act_history.insert(0, torch.tensor(act).to(self.device))
        self.obs_history.insert(0,  torch.tensor(state).to(self.device))
        self.act_history = self.act_history[0:self.history_len]
        self.obs_history = self.obs_history[0:self.history_len]

    def aggregate_history(self):
        curr_hist_delta = self.history_len - len(self.obs_history)
        for i in range(curr_hist_delta):
            self.obs_history.append(torch.zeros(self.obs_shape).to(self.device))
        if self.use_actions:
            obs_and_acts_hist=[]
            curr_hist_delta = self.history_len - len(self.act_history)
            for i in range(curr_hist_delta):
                self.act_history.append(torch.zeros(2).to(self.device))
            obs_history = torch.stack(self.obs_history)
            act_history = torch.stack(self.act_history)
            return torch.cat([obs_history.reshape(self.history_len, -1), act_history], dim=1)
        return torch.stack(self.obs_history).reshape(self.history_len, self.obs_size)

class DQNAgent(BaseAgent):

    def __init__(self,
                 config,
                 dqn_config,
                 device,
                 n_actions,
                 obs_shape):

        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape)
        self.buffer = ReplayMemory(self.buffer_size, "dqn")
        self.eps_init = dqn_config["eps_init"]
        self.eps_final = dqn_config["eps_final"]
        self.eps_decay = dqn_config["eps_decay"]

        self.q_net = construct_model(model_type=self.model_type,
                                     in_size=self.val_obs_size,
                                     out_size=self.n_actions,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     batch_norm=self.batch_norm,
                                     in_channels=self.history_len,
                                     h=self.obs_shape[0],
                                     w=self.obs_shape[1])
        self.t_net = construct_model(model_type=self.model_type,
                                     in_size=self.val_obs_size,
                                     out_size=self.n_actions,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     batch_norm=self.batch_norm,
                                     in_channels=self.history_len,
                                     h=self.obs_shape[0],
                                     w=self.obs_shape[1])

        self.t_net.load_state_dict(self.q_net.state_dict())
        self.q_net.to(self.device)
        self.t_net.to(self.device)

        if self.opt_type == "rmsprop":
            self.optimizer = optim.RMSprop(self.q_net.parameters())

    def select_action(self, state):
        sample = np.random.uniform(0, 1)
        eps_threshold = self.eps_final + (self.eps_init - self.eps_final) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.q_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.randint(0, self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return
        transitions = self.buffer.sample(self.batch_size)
        # Transpose the batch
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None]).float()

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.q_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.t_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # print("loss: ", loss.item())
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()


class A2CAgent(BaseAgent):
    def __init__(self,
                 config,
                 a2c_config,
                 device,
                 n_actions,
                 obs_shape):

        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape,
                           is_pc=a2c_config["is_pc"])
        self.buffer = ReplayMemory(self.buffer_size, "a2c")
        self.q_net = construct_model(model_type=self.model_type,
                                     in_size=self.val_obs_size,
                                     out_size=self.n_actions,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     batch_norm=self.batch_norm,
                                     in_channels=self.history_len,
                                     h=self.obs_shape[0],
                                     w=self.obs_shape[1])
        self.t_net = construct_model(model_type=self.model_type,
                                     in_size=self.val_obs_size,
                                     out_size=self.n_actions,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     batch_norm=self.batch_norm,
                                     in_channels=self.history_len,
                                     h=self.obs_shape[0],
                                     w=self.obs_shape[1])
        self.actor = Actor(self.obs_size, self.n_actions, self.hidden_size, self.temperature)

        self.t_net.load_state_dict(self.q_net.state_dict())
        self.q_net.to(self.device)
        self.t_net.to(self.device)
        self.actor.to(self.device)
        self.transition: list = list()

        if self.opt_type.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(self.q_net.parameters())
            self.actor_optimizer = optim.RMSprop(self.actor.parameters())

    def update_history(self, act, state):
        self.act_history.insert(0, act)
        self.obs_history.insert(0,  torch.tensor(state, requires_grad=False).to(self.device))
        self.act_history = self.act_history[0:self.history_len]
        self.obs_history = self.obs_history[0:self.history_len]

    def aggregate_history(self):
        curr_hist_delta = self.history_len - len(self.obs_history)
        for i in range(curr_hist_delta):
            self.obs_history.append(torch.zeros(self.obs_shape).to(self.device))
        return torch.stack(self.obs_history)

    def select_action(self, state):
        self.steps_done += 1
        dist = self.actor(state.clone().flatten())
        
        return dist

    def optimize_model(self):

        state, log_prob, next_state, reward, done, act = self.transition
        
         # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        next_state = torch.unsqueeze(next_state, dim=0)
        pred_value = self.q_net(state)
        targ_value = self.q_net(next_state)
        dist = self.select_action(state)
        next_dist = self.select_action(next_state)

        expected_curr_value = torch.matmul(pred_value, dist)
        expected_next_value = torch.matmul(self.t_net(next_state), next_dist)

        targ_value =  reward + self.gamma * targ_value * mask
        value_loss = F.smooth_l1_loss(pred_value, targ_value.detach())
        
        # update value
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        # advantage = Q(s_t, a) - V(s_t)
        u_dist = ((1/self.n_actions)*torch.ones(self.n_actions)).to(self.device)
        advantage = (targ_value.data[0][act.item()] - torch.matmul(pred_value, u_dist)).detach()  # not backpropagated
        policy_loss = -advantage * log_prob
        # policy_loss += self.entropy_weight * -log_prob  # entropy maximization

        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()

## A2C with policy-conditioned value function
class A2CPCAgent(BaseAgent):
    def __init__(self,
                 config,
                 a2c_config,
                 device,
                 n_actions,
                 obs_shape):
        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape,
                           is_pc=True)
        self.buffer = ReplayMemory(self.buffer_size, "a2c")
        self.q_net = construct_model(model_type=self.model_type,
                                     in_size=self.val_obs_size,
                                     out_size=self.n_actions,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     batch_norm=self.batch_norm,
                                     in_channels=self.history_len,
                                     h=self.obs_shape[0],
                                     w=self.obs_shape[1])
        self.t_net = construct_model(model_type=self.model_type,
                                     in_size=self.val_obs_size,
                                     out_size=self.n_actions,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     batch_norm=self.batch_norm,
                                     in_channels=self.history_len,
                                     h=self.obs_shape[0],
                                     w=self.obs_shape[1])
        self.actor = Actor(self.obs_size, self.n_actions, self.hidden_size, self.temperature)
        self.o_actor = Actor(self.obs_size, self.n_actions, self.hidden_size, self.temperature)

        self.t_net.load_state_dict(self.q_net.state_dict())
        self.q_net.to(self.device)
        self.t_net.to(self.device)
        self.actor.to(self.device)
        self.o_actor.to(self.device)
        self.transition: list = list()

        if self.opt_type.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(self.q_net.parameters())
            self.actor_optimizer = optim.RMSprop(self.actor.parameters())

    def update_history(self, act, state):
        self.act_history.insert(0, act)
        self.obs_history.insert(0,  torch.tensor(state, requires_grad=False).to(self.device))
        self.act_history = self.act_history[0:self.history_len]
        self.obs_history = self.obs_history[0:self.history_len]

    def aggregate_history(self):
        curr_hist_delta = self.history_len - len(self.obs_history)
        for i in range(curr_hist_delta):
            self.obs_history.append(torch.zeros(self.obs_shape).to(self.device))
        return torch.stack(self.obs_history)

    def select_action(self, state):
        self.steps_done += 1
        dist = self.actor(state.clone().flatten())
        
        return dist

    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return None, None
        transitions = self.buffer.sample(self.batch_size)
        # Transpose the batch
        batch = A2CTransition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None]).float()
        non_final_index = non_final_mask.nonzero()

        num_nfns = non_final_next_states.shape[0]

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        dist_batch = torch.cat(batch.dist)
        state_dist_batch = torch.cat([state_batch.reshape(self.batch_size, self.obs_size), dist_batch], dim=1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.q_net(state_dist_batch).gather(1, action_batch.reshape(self.batch_size, 1))

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        #Not max anymore, Expectation over the current policies instead
        with torch.no_grad():
            next_state_a_policy = self.actor(non_final_next_states.reshape(num_nfns, self.n_actions)).detach()
            next_state_o_policy = self.o_actor(non_final_next_states.reshape(num_nfns, self.n_actions)).detach()
            non_final_next_states_dists = torch.cat([non_final_next_states.reshape(num_nfns, self.obs_size), next_state_o_policy], dim=1)

            next_state_values[non_final_mask] = torch.bmm(self.t_net(non_final_next_states_dists).reshape(num_nfns, 1, self.n_actions), 
                                                          next_state_a_policy.reshape(num_nfns, self.n_actions, 1)).squeeze()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        value_loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1).detach())

        self.optimizer.zero_grad()
        value_loss.backward()
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        state, log_p, next_state, reward, done, act, dist = self.transition
        a_dist = self.select_action(state)
        log_prob = torch.log(a_dist[act])
        mask = 1 - done
        pred_value = self.q_net(torch.cat([state.reshape(1, self.obs_size), dist], dim=1))
        targ_value = self.q_net(torch.cat([next_state.reshape(1, self.obs_size), self.o_actor(next_state.reshape(1, self.obs_size))], dim=1))
        targ_value =  reward + self.gamma * targ_value * mask
        
        # advantage = Q(s_t, a) - V(s_t)
        advantage = (targ_value.data[0][act.item()] - torch.mean(pred_value)).detach()  # not backpropagated
        policy_loss = -advantage * log_prob

        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        return policy_loss.item(), value_loss.item()


class NashActorCriticAgent(BaseAgent):

    def __init__(self,
                 config,
                 nash_ac_config,
                 sgd_config,
                 device,
                 n_actions,
                 obs_shape):
        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape,
                           is_nash=True)
        self.q_head_params = []
        self.surr_q_head_params = []
        self.actor_in_size = self.history_len * self.act_obs_size
        if self.model_type == "lstm":
            self.out_size = nash_ac_config["lstm_out"]
            self.q_head = LinearQHead(self.out_size + 2 * self.n_actions, 1)
            self.surr_q_head = LinearQHead(self.out_size + 2 * self.n_actions, 1)
            self.q_head.to(self.device)
            self.surr_q_head.to(self.device)
            self.q_head_params = list(self.q_head.parameters())
            self.surr_q_head_params = list(self.surr_q_head.parameters())
        else:
            self.out_size = 1

        self.buffer = ReplayMemory(self.buffer_size, "nash_ac")
        self.q_net = construct_model(model_type=self.model_type,
                                     in_size=self.val_obs_size,
                                     out_size=self.out_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     batch_norm=self.batch_norm,
                                     in_channels=self.history_len,
                                     h=self.obs_shape[0],
                                     w=self.obs_shape[1])
        self.t_net = construct_model(model_type=self.model_type,
                                     in_size=self.val_obs_size,
                                     out_size=self.out_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     batch_norm=self.batch_norm,
                                     in_channels=self.history_len,
                                     h=self.obs_shape[0],
                                     w=self.obs_shape[1])
        self.surr_q_net = construct_model(model_type=self.model_type,
                                          in_size=self.val_obs_size,
                                          out_size=self.out_size,
                                          hidden_size=self.hidden_size,
                                          num_layers=self.num_layers,
                                          batch_norm=self.batch_norm,
                                          in_channels=self.history_len,
                                          h=self.obs_shape[0],
                                          w=self.obs_shape[1])
        self.actor = Actor(self.actor_in_size, self.n_actions, self.hidden_size, self.temperature)
        self.o_actor = Actor(self.actor_in_size, self.n_actions, self.hidden_size, self.temperature)

        self.t_net.load_state_dict(self.q_net.state_dict())
        self.q_net.to(self.device)
        self.t_net.to(self.device)
        self.surr_q_net.to(self.device)
        self.actor.to(self.device)
        self.o_actor.to(self.device)
        self.transition: list = list()

        # Value optimizers
        if self.opt_type.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(list(self.q_net.parameters()) + list(self.q_head_params))
        
        # Joint surrogate Q-value function and policy optimizer
        self.actor_optimizer = optim.SGD(list(self.surr_q_net.parameters()) + 
                                         list(self.actor.parameters()) + list(self.surr_q_head_params), 
                                         lr=sgd_config["lr"],
                                         momentum=sgd_config["momentum"],
                                         maximize=True)

    def select_action(self, state):
        self.steps_done += 1
        dist = self.actor(state.clone().flatten())
        
        return dist

    def lstm_q_net_forward(self, state_batch, a_batch, b_batch, is_surr):
        lstm_out_vals = self.q_net(state_batch)
        state_a_b_batch = torch.cat([lstm_out_vals, a_batch, b_batch], dim=1)
        if is_surr:
            state_action_values = self.surr_q_head(state_a_b_batch)
        else:
            state_action_values = self.q_head(state_a_b_batch)

        return state_action_values


    def optimize_model(self, cum_steps):
        if len(self.buffer) < self.batch_size:
            return None, None
        transitions = self.buffer.sample(self.batch_size)
        # Transpose the batch
        batch = NashACTransition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None]).float()
        non_final_index = non_final_mask.nonzero()

        num_nfns = non_final_next_states.shape[0]

        state_batch = torch.cat(batch.state)
        a_batch = torch.cat(batch.a)
        b_batch = torch.cat(batch.b)
        reward_batch = torch.cat(batch.reward)

        one_hot_a_batch = torch.zeros((self.batch_size, self.n_actions)).to(self.device)
        one_hot_b_batch = torch.zeros((self.batch_size, self.n_actions)).to(self.device)
        one_hot_a_batch = one_hot_a_batch.scatter(1, a_batch.reshape(self.batch_size, 1), 1)
        one_hot_b_batch = one_hot_b_batch.scatter(1, b_batch.reshape(self.batch_size, 1), 1)

        # Compute Q(s_t, a, b)
        if self.model_type == "lstm":
            state_action_values = self.lstm_q_net_forward(state_batch, one_hot_a_batch, one_hot_b_batch, False)
        else:
            state_a_b_batch = torch.cat([state_batch.reshape(self.batch_size, self.obs_size), one_hot_a_batch, one_hot_b_batch], dim=1)
            state_action_values = self.q_net(state_a_b_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        #Not max anymore, Expectation over the current policies instead. Approximated via sampling
        with torch.no_grad():
            next_state_a_policy = self.actor(non_final_next_states.reshape(num_nfns, self.actor_in_size)).detach()
            next_state_b_policy = self.o_actor(non_final_next_states.reshape(num_nfns, self.actor_in_size)).detach()

            # Independent sampling for a', b'
            cum_next_state_a_policy = torch.cumsum(next_state_a_policy, dim=1)
            cum_next_state_b_policy = torch.cumsum(next_state_b_policy, dim=1)
            u_dist_a = torch.rand((self.batch_size, self.n_actions)).to(self.device)
            u_dist_b = torch.rand((self.batch_size, self.n_actions)).to(self.device)
            a_prime = torch.argmax((u_dist_a < cum_next_state_a_policy).long(), dim=1)
            b_prime = torch.argmax((u_dist_b < cum_next_state_b_policy).long(), dim=1)

            # One-hot encoding of a', b'
            one_hot_a_prime_batch = torch.zeros((self.batch_size, self.n_actions)).to(self.device)
            one_hot_b_prime_batch = torch.zeros((self.batch_size, self.n_actions)).to(self.device)
            one_hot_a_prime_batch = one_hot_a_prime_batch.scatter(1, a_prime.reshape(self.batch_size, 1), 1)
            one_hot_b_prime_batch = one_hot_b_prime_batch.scatter(1, b_prime.reshape(self.batch_size, 1), 1)
            
            if self.model_type == "lstm":
                next_state_values[non_final_mask] = self.lstm_q_net_forward(non_final_next_states, 
                                                                            one_hot_a_prime_batch, 
                                                                            one_hot_b_prime_batch, 
                                                                            False).squeeze()
            else:
                non_final_next_states_and_acts = torch.cat([non_final_next_states.reshape(num_nfns, self.obs_size), one_hot_a_prime_batch, 
                                                           one_hot_b_prime_batch], dim=1)
                next_state_values[non_final_mask] = self.t_net(non_final_next_states_and_acts).squeeze()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        value_loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1).detach())

        self.optimizer.zero_grad()
        value_loss.backward()
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Set surrogate weights to Q-value function's weights
        self.surr_q_net.load_state_dict(self.t_net.state_dict())

        if cum_steps%100 == 0:
            state, dist_b = self.transition
            dist_a = self.select_action(state)
            if self.model_type == "lstm":
                self.surr_q_head.load_state_dict(self.q_head.state_dict())
                game_value = self.lstm_q_net_forward(state, 
                                                    dist_a.reshape(1, self.n_actions), 
                                                    dist_b.reshape(1, self.n_actions), 
                                                    True)
            else:
                state_a_b = torch.cat([state.reshape(1, self.obs_size), dist_a.reshape(1, self.n_actions), 
                                    dist_b.reshape(1, self.n_actions)], dim=1)
                game_value = self.surr_q_net(state_a_b)

            # Use surrogate Q^A(s, a, b), to optimize \pi^A(s)
            self.actor_optimizer.zero_grad()
            game_value.backward()
            self.actor_optimizer.step()
        else:
            game_value = torch.zeros((1))
        
        return game_value.item(), value_loss.item()


class ReinforcedNashActorCriticAgent(BaseAgent):

    def __init__(self,
                 config,
                 rf_nash_ac_config,
                 sgd_config,
                 device,
                 n_actions,
                 obs_shape,
                 is_agent=False):
        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape,
                           is_rf_nash=True)
        self.is_agent = is_agent
        self.q_head_params = []
        self.surr_q_head_params = []
        self.actor_in_size = self.history_len * self.act_obs_size
        self.policy_hist_len = rf_nash_ac_config["policy_hist_len"]
        self.is_p_pc = rf_nash_ac_config["is_p_pc"] and not self.is_agent

        self.num_hidden = 0 if self.is_p_pc else 1
        self.actor_in_size = self.actor_in_size + self.n_actions if self.is_p_pc else self.actor_in_size
        self.o_actor_in_size = self.history_len * self.act_obs_size if self.is_p_pc else self.actor_in_size + self.n_actions

        if self.model_type == "lstm":
            self.out_size = rf_nash_ac_config["lstm_out"]
            self.q_head = LinearQHead(self.out_size + 2 * self.n_actions, 1)
            self.surr_q_head = LinearQHead(self.out_size + 2 * self.n_actions, 1)
            self.q_head.to(self.device)
            self.surr_q_head.to(self.device)
            self.q_head_params = list(self.q_head.parameters())
            self.surr_q_head_params = list(self.surr_q_head.parameters())
        else:
            self.out_size = 1

        self.buffer = ReplayMemory(self.buffer_size, "rf_nash_ac")
        self.policy_buffer = ReplayMemory(self.buffer_size, "rf_nash_ac")
        self.q_net = construct_model(model_type=self.model_type,
                                     in_size=self.val_obs_size,
                                     out_size=self.out_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     batch_norm=self.batch_norm,
                                     in_channels=self.history_len,
                                     h=self.obs_shape[0],
                                     w=self.obs_shape[1])
        self.t_net = construct_model(model_type=self.model_type,
                                     in_size=self.val_obs_size,
                                     out_size=self.out_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     batch_norm=self.batch_norm,
                                     in_channels=self.history_len,
                                     h=self.obs_shape[0],
                                     w=self.obs_shape[1])
        self.surr_q_net = construct_model(model_type=self.model_type,
                                          in_size=self.val_obs_size,
                                          out_size=self.out_size,
                                          hidden_size=self.hidden_size,
                                          num_layers=self.num_layers,
                                          batch_norm=self.batch_norm,
                                          in_channels=self.history_len,
                                          h=self.obs_shape[0],
                                          w=self.obs_shape[1])
        self.actor = Actor(self.actor_in_size, self.n_actions, self.hidden_size, self.temperature, 0)
        self.o_actor = Actor(self.o_actor_in_size, self.n_actions, self.hidden_size, self.temperature, 0)

        self.t_net.load_state_dict(self.q_net.state_dict())
        self.q_net.to(self.device)
        self.t_net.to(self.device)
        self.surr_q_net.to(self.device)
        self.actor.to(self.device)
        self.o_actor.to(self.device)
        self.transition: list = list()
        self.model = CoinGame(2, 1)

        # Value optimizers
        if self.opt_type.lower() == "sgd":
            self.optimizer = optim.RMSprop(list(self.q_net.parameters()) + list(self.q_head_params))
            self.actor_optimizer = optim.SGD(list(self.surr_q_net.parameters()) + 
                                         list(self.actor.parameters()) + list(self.surr_q_head_params), 
                                         lr=sgd_config["lr"],
                                         momentum=sgd_config["momentum"],
                                         maximize=True)
            
        elif self.opt_type.lower() == "adam":
            self.optimizer = optim.Adam(list(self.q_net.parameters()) + list(self.q_head_params),
                                        lr=sgd_config["lr"])
            # Joint surrogate Q-value function and policy optimizer
            self.actor_optimizer = optim.Adam(list(self.surr_q_net.parameters()) + 
                                              list(self.actor.parameters()) + list(self.surr_q_head_params), 
                                              lr=sgd_config["lr"],
                                              maximize=True)

    def select_action(self, state, dist_b=None):
        self.steps_done += 1
        if self.is_p_pc:
            dist = self.actor(torch.cat([state.flatten(), dist_b]))
        else:
            dist = self.actor(state.flatten())

        return dist

    def lstm_q_net_forward(self, state_batch, a_batch, b_batch, is_surr):
        if is_surr:
            lstm_out_vals = self.surr_q_net(state_batch)
        else:
            lstm_out_vals = self.q_net(state_batch)
        state_a_b_batch = torch.cat([lstm_out_vals, a_batch, b_batch], dim=1)
        if is_surr:
            state_action_values = self.surr_q_head(state_a_b_batch)
        else:
            state_action_values = self.q_head(state_a_b_batch)

        return state_action_values

    def optimize_model(self):
        # TODO: Reinforce estimation to optimize the policy (Could use soft actions as well)
        if len(self.buffer) < self.batch_size + self.policy_hist_len:
            return None, None
        transitions = self.buffer.sample(self.batch_size)

        # Transpose the batch
        batch = RfNashACTransition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        a_batch = torch.cat(batch.a)
        b_batch = torch.cat(batch.b)
        reward_batch = torch.cat(batch.reward)

        # Compute r(s_t, a_t, b_t)
        if self.model_type == "lstm":
            state_action_values = self.lstm_q_net_forward(state_batch, a_batch, b_batch, False)
        else:
            state_a_b_batch = torch.cat([state_batch.reshape(self.batch_size, self.obs_size), a_batch, b_batch], dim=1)
            state_action_values = self.q_net(state_a_b_batch)

        # Compute the expected reward values
        expected_state_action_values = reward_batch

        # Want to estimate mean instead of median
        criterion = nn.MSELoss()
        value_loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1).detach())

        self.optimizer.zero_grad()
        value_loss.backward()
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Set surrogate weights to Q-value function's weights
        self.surr_q_net.load_state_dict(self.t_net.state_dict())
        self.surr_q_head.load_state_dict(self.q_head.state_dict())

        # Use surrogate r^A(s, a, b), to optimize \pi^A(s)
        state, dist_b = self.transition

        estimated_rewards = []
        self.model.grid = state[0, 0, 0:self.obs_size].reshape(1, self.obs_size).cpu().detach().numpy()
        # Use surrogate r^A(s, a, b), and monte-carlo rollout to optimize \pi^A(s)
        for i in range(self.policy_hist_len):
            last_state = state[:, 1: , :].reshape(1, self.history_len * self.obs_size).cpu().detach().numpy()
            if self.is_agent:
                dist_a = self.actor(state.flatten())
                dist_b = self.o_actor(torch.cat([state.flatten(), dist_a.flatten()]))
            else:
                dist_b = self.o_actor(state.flatten())
                dist_a = self.actor(torch.cat([state.flatten(), dist_b.flatten()]))

            action_a = np.array([np.random.choice(self.n_actions, p=dist_a.cpu().detach().numpy())])
            action_b = np.array([np.random.choice(self.n_actions, p=dist_b.cpu().detach().numpy())])
            
            curr_state, rewards, done, _, _ = self.model.step((action_a, action_b))
            curr_state = np.concatenate([curr_state, action_a.reshape(1, 1), action_b.reshape(1, 1)], axis=1)

            state = torch.tensor(np.expand_dims(np.concatenate([curr_state ,last_state]), 0), 
                                 requires_grad=False, 
                                 device=self.device,
                                 dtype=torch.float32)

            if self.model_type == "lstm":
                estimated_reward = self.lstm_q_net_forward(state, 
                                                           dist_a.reshape(1, self.n_actions), 
                                                           dist_b.reshape(1, self.n_actions), 
                                                           True)
            else:
                state_a_b_batch = torch.cat([state.reshape(1, self.obs_size * self.history_len), 
                                            dist_a.reshape(1, self.n_actions), 
                                            dist_b.reshape(1, self.n_actions)], 
                                            dim=1)
                estimated_reward = self.q_net(state_a_b_batch)

            a_t_prob = torch.take(dist_a, torch.tensor(action_a, requires_grad=False, device=self.device))
            b_t_prob = torch.take(dist_b, torch.tensor(action_b, requires_grad=False, device=self.device))

            estimated_reward = estimated_reward * a_t_prob * b_t_prob
            estimated_rewards.append(estimated_reward)

        game_value = torch.sum(torch.cat(estimated_rewards, dim=0))/self.policy_hist_len

        self.actor_optimizer.zero_grad()
        game_value.backward()
        self.actor_optimizer.step()
        
        # detach is faster
        return game_value.item(), value_loss.item()

class SRNashActorCriticAgent(BaseAgent):

    def __init__(self,
                 config,
                 rf_nash_ac_config,
                 sgd_config,
                 device,
                 n_actions,
                 obs_shape):
        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape,
                           is_rf_nash=True)
        self.q_head_params = []
        self.surr_q_head_params = []
        self.actor_in_size = self.history_len * self.act_obs_size
        self.policy_hist_len = rf_nash_ac_config["policy_hist_len"]
        self.is_p_pc = rf_nash_ac_config["is_p_pc"]

        self.num_hidden = 0
        self.actor_in_size = self.actor_in_size + self.n_actions if self.is_p_pc else self.actor_in_size
        self.o_actor_in_size = self.actor_in_size

        if self.model_type == "lstm":
            self.out_size = rf_nash_ac_config["lstm_out"]
            self.q_head = LinearQHead(self.out_size + 2 * self.n_actions, 1)
            self.surr_q_head = LinearQHead(self.out_size + 2 * self.n_actions, 1)
            self.q_head.to(self.device)
            self.surr_q_head.to(self.device)
            self.q_head_params = list(self.q_head.parameters())
            self.surr_q_head_params = list(self.surr_q_head.parameters())
        else:
            self.out_size = 1

        self.buffer = ReplayMemory(self.buffer_size, "rf_nash_ac")
        self.q_net = construct_model(model_type=self.model_type,
                                     in_size=self.val_obs_size,
                                     out_size=self.out_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     batch_norm=self.batch_norm,
                                     in_channels=self.history_len,
                                     h=self.obs_shape[0],
                                     w=self.obs_shape[1])
        self.t_net = construct_model(model_type=self.model_type,
                                     in_size=self.val_obs_size,
                                     out_size=self.out_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     batch_norm=self.batch_norm,
                                     in_channels=self.history_len,
                                     h=self.obs_shape[0],
                                     w=self.obs_shape[1])
        self.surr_q_net = construct_model(model_type=self.model_type,
                                          in_size=self.val_obs_size,
                                          out_size=self.out_size,
                                          hidden_size=self.hidden_size,
                                          num_layers=self.num_layers,
                                          batch_norm=self.batch_norm,
                                          in_channels=self.history_len,
                                          h=self.obs_shape[0],
                                          w=self.obs_shape[1])
        self.actor = Actor(self.actor_in_size, self.n_actions, self.hidden_size, self.temperature, 0)
        self.o_actor = Actor(self.o_actor_in_size, self.n_actions, self.hidden_size, self.temperature, 0)

        self.t_net.load_state_dict(self.q_net.state_dict())
        self.q_net.to(self.device)
        self.t_net.to(self.device)
        self.surr_q_net.to(self.device)
        self.actor.to(self.device)
        self.o_actor.to(self.device)
        self.transition: list = list()
        self.model = CoinGame(2, 1)

        # Value optimizers
        if self.opt_type.lower() == "sgd":
            self.optimizer = optim.RMSprop(list(self.q_net.parameters()) + list(self.q_head_params))
            self.actor_optimizer = optim.SGD(list(self.surr_q_net.parameters()) + 
                                         list(self.actor.parameters()) + list(self.surr_q_head_params), 
                                         lr=sgd_config["lr"],
                                         momentum=sgd_config["momentum"],
                                         maximize=True)
            
        elif self.opt_type.lower() == "adam":
            self.optimizer = optim.Adam(list(self.q_net.parameters()) + list(self.q_head_params),
                                        lr=sgd_config["lr"])
            # Joint surrogate Q-value function and policy optimizer
            self.actor_optimizer = optim.Adam(list(self.surr_q_net.parameters()) + 
                                              list(self.actor.parameters()) + list(self.surr_q_head_params), 
                                              lr=sgd_config["lr"],
                                              maximize=True)

    def select_action(self, state, dist_b=None):
        self.steps_done += 1
        dist = self.actor(torch.cat([state.flatten(), -1 * torch.ones(self.n_actions).to(self.device)]))

        return dist

    def lstm_q_net_forward(self, state_batch, a_batch, b_batch, is_surr):
        if is_surr:
            lstm_out_vals = self.surr_q_net(state_batch)
        else:
            lstm_out_vals = self.q_net(state_batch)
        state_a_b_batch = torch.cat([lstm_out_vals, a_batch, b_batch], dim=1)
        if is_surr:
            state_action_values = self.surr_q_head(state_a_b_batch)
        else:
            state_action_values = self.q_head(state_a_b_batch)

        return state_action_values

    def optimize_model(self, is_inducer):
        # TODO: Reinforce estimation to optimize the policy (Could use soft actions as well)
        if len(self.buffer) < self.batch_size:
            return None, None
        transitions = self.buffer.sample(self.batch_size)

        # Transpose the batch
        batch = RfNashACTransition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        a_batch = torch.cat(batch.a)
        b_batch = torch.cat(batch.b)
        reward_batch = torch.cat(batch.reward)

        # Compute r(s_t, a_t, b_t)
        if self.model_type == "lstm":
            state_action_values = self.lstm_q_net_forward(state_batch, a_batch, b_batch, False)
        else:
            state_a_b_batch = torch.cat([state_batch.reshape(self.batch_size, self.obs_size), a_batch, b_batch], dim=1)
            state_action_values = self.q_net(state_a_b_batch)

        # Compute the expected reward values
        expected_state_action_values = reward_batch

        # Want to estimate mean instead of median
        criterion = nn.MSELoss()
        value_loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1).detach())

        self.optimizer.zero_grad()
        value_loss.backward()
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Set surrogate weights to Q-value function's weights
        self.surr_q_net.load_state_dict(self.t_net.state_dict())
        self.surr_q_head.load_state_dict(self.q_head.state_dict())

        # Use surrogate r^A(s, a, b), to optimize \pi^A(s)
        state, dist_b = self.transition
        
        estimated_rewards = []
        self.model.grid = state[0, 0, 0:self.obs_size].reshape(1, self.obs_size).cpu().detach().numpy()
        # Use surrogate r^A(s, a, b), and monte-carlo rollout to optimize \pi^A(s)
        for i in range(self.policy_hist_len):
            last_state = state[:, 1: , :].reshape(1, self.history_len * self.obs_size).cpu().detach().numpy()
            if is_inducer:
                dist_a = self.actor(torch.cat([state.flatten(), -1 * torch.ones(self.n_actions).to(self.device)]))
                dist_b = self.o_actor(torch.cat([state.flatten(), dist_a.flatten()]))
            else:
                dist_b = self.o_actor(torch.cat([state.flatten(), -1 * torch.ones(self.n_actions).to(self.device)]))
                dist_a = self.actor(torch.cat([state.flatten(), dist_b.flatten()]))

            action_a = np.array([np.random.choice(self.n_actions, p=dist_a.cpu().detach().numpy())])
            action_b = np.array([np.random.choice(self.n_actions, p=dist_b.cpu().detach().numpy())])
            
            curr_state, rewards, done, _, _ = self.model.step((action_a, action_b))
            curr_state = np.concatenate([curr_state, action_a.reshape(1, 1), action_b.reshape(1, 1)], axis=1)

            state = torch.tensor(np.expand_dims(np.concatenate([curr_state ,last_state]), 0), 
                                 requires_grad=False, 
                                 device=self.device,
                                 dtype=torch.float32)

            if self.model_type == "lstm":
                estimated_reward = self.lstm_q_net_forward(state, 
                                                           dist_a.reshape(1, self.n_actions), 
                                                           dist_b.reshape(1, self.n_actions), 
                                                           True)
            else:
                state_a_b_batch = torch.cat([state.reshape(1, self.obs_size * self.history_len), 
                                            dist_a.reshape(1, self.n_actions), 
                                            dist_b.reshape(1, self.n_actions)], 
                                            dim=1)
                estimated_reward = self.q_net(state_a_b_batch)

            a_t_prob = torch.take(dist_a, torch.tensor(action_a, requires_grad=False, device=self.device))
            b_t_prob = torch.take(dist_b, torch.tensor(action_b, requires_grad=False, device=self.device))

            estimated_reward = estimated_reward * a_t_prob * b_t_prob
            estimated_rewards.append(estimated_reward)

        game_value = torch.sum(torch.cat(estimated_rewards, dim=0))/self.policy_hist_len

        self.actor_optimizer.zero_grad()
        game_value.backward()
        self.actor_optimizer.step()
        
        # detach is faster
        return game_value.item(), value_loss.item()