import jax
import equinox as eqx
from typing import List
import distrax

class ActorNetwork(eqx.Module):
    """Actor network"""

    layers: list

    def __init__(self, key, in_shape, hidden_features: List[int], num_actions):
        keys = jax.random.split(key, len(hidden_features))
        self.layers = [
            eqx.nn.Linear(in_shape, hidden_features[0], key=keys[0])
        ]
        for i, feature in enumerate(hidden_features[:-1]):
            self.layers.append(eqx.nn.Linear(feature, hidden_features[i+1], key=keys[i]))
        self.layers.append(eqx.nn.Linear(hidden_features[-1], num_actions, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return distrax.Categorical(logits=self.layers[-1](x))
    
        # def f(carry, input):
        #     return jax.nn.relu(carry(input)), None
        
        # out, _ = jax.lax.scan(f, self.layers, x)
        # return distrax.Categorical(logits=out)
    
class CriticNetwork(eqx.Module):
    """
        Critic network with a single output
        Used for example to output V when given a state
        or Q when given a state and action
    """
    layers: list

    def __init__(self, key, in_shape, hidden_layers: List[int]):
        keys = jax.random.split(key, len(hidden_layers))
        self.layers = [ # init with first layer
            eqx.nn.Linear(in_shape, hidden_layers[0], key=keys[0])
        ]
        for i, feature in enumerate(hidden_layers[:-1]):
            self.layers.append(eqx.nn.Linear(feature, hidden_layers[i+1], key=keys[i]))
        self.layers.append(eqx.nn.Linear(hidden_layers[-1], 1, key=keys[-1]))

class Q_CriticNetwork(eqx.Module):
    """'
        Critic network that outputs values for each action
        e.g. a list of Q-values
    """

    layers: list

    def __init__(self, key, in_shape, hidden_layers: List[int], num_actions):
        keys = jax.random.split(key, len(hidden_layers))
        self.layers = [
            eqx.nn.Linear(in_shape, hidden_layers[0], key=keys[0])
        ]
        for i, feature in enumerate(hidden_layers[:-1]):
            self.layers.append(eqx.nn.Linear(feature, hidden_layers[i+1], key=keys[i]))
        self.layers.append(eqx.nn.Linear(hidden_layers[-1], num_actions, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x)
    
def create_actor_critic_network(
        key,
        in_shape: int,
        actor_features: List[int],
        critic_features: List[int],
        num_env_actions: int,
    ):
    """
        Create actor and critic networks
    """
    actor_key, critic_key = jax.random.split(key, 2)
    actor = ActorNetwork(actor_key, in_shape, actor_features, num_env_actions)
    critic = CriticNetwork(critic_key, in_shape, critic_features)
    return actor, critic

def create_actor_critic_critic_network(
        key,
        in_shape: int,
        actor_features: List[int],
        critic_features: List[int],
        num_env_actions: int,
    ):
    """
        Create actor, 2 critics and target networks (e.g. for SAC)
    """
    actor_key, critic1_key, critic2_key = jax.random.split(key, 3)
    actor = ActorNetwork(actor_key, in_shape, actor_features, num_env_actions)
    critic = Q_CriticNetwork(critic1_key, in_shape, critic_features, num_env_actions)
    critic2 = Q_CriticNetwork(critic2_key, in_shape, critic_features, num_env_actions)
    critic1_target = jax.tree_map(lambda x: x, critic)
    critic2_target = jax.tree_map(lambda x: x, critic2)
    return actor, critic, critic2, critic1_target, critic2_target