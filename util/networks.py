import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, List

class SharedNetwork(nn.Module):
    """Shared network"""

    features: List[int]
    activation: Callable = nn.tanh
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        for i, feature in enumerate(self.features):
            x = nn.Dense(
                feature, 
                name=f"fc{i}",
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)), 
                bias_init=nn.initializers.constant(0.0)
            )(x)
            x = self.activation(x)
        return x


class ActorNetwork(nn.Module):
    """Actor network"""

    features: List[int]
    num_actions: int
    activation: Callable = nn.tanh
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        for i, feature in enumerate(self.features):
            x = nn.Dense(
                feature, 
                name=f"fc{i}",
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)), 
                bias_init=nn.initializers.constant(0.0)
            )(x)
            x = self.activation(x)
        logits = nn.Dense(
            self.num_actions, 
            name="logits",
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0)
            )(x)
        return logits


class CriticNetwork(nn.Module):
    """Critic network"""

    features: List[int]
    activation: Callable = nn.tanh
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        for i, feature in enumerate(self.features):
            x = nn.Dense(
                feature, 
                name=f"fc{i}",
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                bias_init=nn.initializers.constant(0.0)
                )(x)
            x = self.activation(x)
        value = nn.Dense(
            1, 
            name="value",
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0)
            )(x)
        return jnp.squeeze(value, axis=-1)
    
def create_actor_critic_network(
        shared_features: List[int],
        actor_features: List[int],
        critic_features: List[int],
        num_env_actions: int
    ):
    """
        Create actor and critic networks
    """
    shared = SharedNetwork(shared_features)
    actor = ActorNetwork(actor_features, num_env_actions)
    critic = CriticNetwork(critic_features)
    return shared, actor, critic