import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import math
from typing import List, Optional, Sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """MLP Generic class."""
    def __init__(self, in_dim: int,
                 out_dim: int,
                 hidden_units: Optional[Sequence[int]] = (256, 256),
                 dropout_rate: Optional[float] = None,
                 use_batch_norm: bool = False,
                 use_dense: bool = True,
                 activation: nn.Module = nn.ReLU(),
                 ):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.feature_size = hidden_units[-1]
        self.activation = activation
        self.use_dense = use_dense

        in_units = [in_dim] + list(hidden_units)
        out_units = list(hidden_units) + [out_dim]
        self._fcs = nn.ModuleList()
        self._bns = nn.ModuleList()
        self._dropouts = nn.ModuleList()
        for i, (in_unit, out_unit) in enumerate(zip(in_units, out_units)):
            if self.use_dense and i > 0:
                in_unit += in_dim
            self._fcs.append(nn.Linear(in_unit, out_unit))
            if use_batch_norm:
                self._bns.append(nn.BatchNorm1d(out_unit))
            if dropout_rate is not None:
                self._dropouts.append(nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, fc in enumerate(self._fcs[:-1]):
            if self.use_dense and i > 0:
                h = torch.cat([h, x], dim=1)
            h = self.activation(fc(h))
            if self.use_batch_norm:
                h = self._bns[i](h)
            if self.dropout_rate is not None:
                h = self._dropouts[i](h)
        if self.use_dense:
            h = torch.cat([h, x], dim=1)
        return self._fcs[-1](h)


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, action_dim, log_std_bounds=(-10, 2), hidden=(256, 256)):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.trunk = MLP(in_dim=obs_dim, out_dim=action_dim, hidden_units=hidden)
        var_size = {"spherical": 1, "diagonal": action_dim}["spherical"]
        self.var_param = nn.Parameter(torch.tensor(np.broadcast_to(0, var_size), dtype=torch.float))

    def forward(self, obs):
        mu = self.trunk(obs)
        log_std_min, log_std_max = self.log_std_bounds
        log_stds = torch.clip(self.var_param, log_std_min, log_std_max)
        dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.exp(log_stds)), 1)
        return dist


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden=(256, 256)):
        super(Actor, self).__init__()
        self.net = MLP(in_dim=state_dim, out_dim=action_dim, hidden_units=hidden)
        self.max_action = max_action

    def forward(self, state):
        a = self.net(state)
        return self.max_action * torch.tanh(a)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(256, 256)):
        super(Critic, self).__init__()
        self.q1 = MLP(in_dim=state_dim + action_dim, out_dim=1, hidden_units=hidden)
        self.q2 = MLP(in_dim=state_dim + action_dim, out_dim=1, hidden_units=hidden)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2


class Value(nn.Module):
    def __init__(self, state_dim, hidden=(256, 256)):
        super(Value, self).__init__()
        self.net = MLP(in_dim=state_dim, out_dim=1, hidden_units=hidden)

    def forward(self, state):
        v = self.net(state)
        return v


def expectile_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class IQL(object):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_action: float,
            discount: float = 0.99,
            tau: float = 0.005,
            expectile: float = 0.8,
            beta: float = 3.0,
            max_weight: float = 100.0,
            deterministic_policy: bool = False,
            normalize_advantage: bool = False,
            normalize_actor_loss: bool = False,
            actor_hidden: Optional[Sequence[int]] = (256, 256),
            critic_hidden: Optional[Sequence[int]] = (256, 256),
            value_hidden: Optional[Sequence[int]] = (256, 256),
    ):
        self.deterministic_policy = deterministic_policy
        if self.deterministic_policy:
            self.actor = Actor(state_dim, action_dim, max_action, hidden=actor_hidden).to(device)
        else:
            self.actor = DiagGaussianActor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        # self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=100)

        self.critic = Critic(state_dim, action_dim, hidden=critic_hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.value = Value(state_dim, hidden=value_hidden).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.expectile = expectile
        self.beta = beta
        self.max_weight = max_weight
        self.tau = tau
        self.normalize_advantage = normalize_advantage
        self.normalize_actor_loss = normalize_actor_loss

        self.total_it = 0

    def select_action(self, state):
        self.actor.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        out = self.actor(state)
        if not self.deterministic_policy:
            out = out.sample()
        action = torch.clip(out, -1, 1) * self.max_action
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        self.actor.train()
        self.critic.train()
        self.value.train()
        self.actor_target.eval()

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute value loss
        with torch.no_grad():
            self.critic_target.eval()
            target_q1, target_q2 = self.critic_target(state, action)
            target_q = torch.min(target_q1, target_q2)
        value = self.value(state)
        value_loss = expectile_loss(target_q - value, expectile=self.expectile).mean()

        # Compute critic loss
        with torch.no_grad():
            self.value.eval()
            next_v = self.value(next_state)
        target_q = reward + not_done * self.discount * next_v
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = ((current_q1 - target_q)**2).mean() + ((current_q2 - target_q)**2).mean()

        # Compute actor loss
        advantage = torch.min(target_q1, target_q2) - value.detach()
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        weights = torch.exp(advantage * self.beta)
        weights = torch.clip(weights, max=self.max_weight)
        if self.deterministic_policy:
            pi = self.actor(state)
            actor_loss = (weights * (pi - action) ** 2).mean()
        else:
            dist = self.actor(state)
            pi = dist.log_prob(action)
            actor_loss = -(weights * pi)
            if self.normalize_actor_loss:
                actor_loss = actor_loss / actor_loss.detach().abs().mean()
            actor_loss = actor_loss.mean()

        # Optimize the value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        info = {'value': value.mean().cpu().detach().item(),
                'q_val': current_q1.mean().cpu().detach().item(),
                'value_loss': value_loss.detach().cpu().numpy().item(),
                'critic_loss': critic_loss.detach().cpu().numpy().item(),
                'actor_loss': actor_loss.detach().cpu().numpy().item()}

        return info

    def save(self, filename):
        torch.save(self.value.state_dict(), filename + "_value")
        torch.save(self.value.state_dict(), filename + "_value_optimizer")

        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.value.load_state_dict(torch.load(filename + "_value"))
        self.value_optimizer.load_state_dict(torch.load(filename + "_value_optimizer"))

        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
