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
            x = jax.nn.relu(layer(x))
        return distrax.Categorical(logits=self.layers[-1](x))
    
        # def f(carry, input):
        #     return jax.nn.relu(carry(input)), None
        
        # out, _ = jax.lax.scan(f, self.layers, x)
        # return distrax.Categorical(logits=out)
    


class Q_CriticNetwork(eqx.Module):
    """Critic network"""

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
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

def create_actor_critic_critic_network(
        key,
        in_shape: int,
        actor_features: List[int],
        critic_features: List[int],
        num_env_actions: int,
    ):
    """
        Create actor and critic critic networks
    """
    actor_key, critic1_key, critic2_key = jax.random.split(key, 3)
    actor = ActorNetwork(actor_key, in_shape, actor_features, num_env_actions)
    critic = Q_CriticNetwork(critic1_key, in_shape, critic_features, num_env_actions)
    critic1_target = Q_CriticNetwork(critic1_key, in_shape, critic_features, num_env_actions) # same key!
    critic2 = Q_CriticNetwork(critic2_key, in_shape, critic_features, num_env_actions)
    critic2_target = Q_CriticNetwork(critic2_key, in_shape, critic_features, num_env_actions) # same key!
    return actor, critic, critic2, critic1_target, critic2_target