# A clean single file implementation of PPO in Equinox
# Largely inspired by cleanRL (https://github.com/vwxyzjn/cleanrl) 
# and PureJaxRL (https://github.com/luchris429/purejaxrl)

import jax
import jax.numpy as jnp
import equinox as eqx
import chex
import gymnasium as gym
import optax
import gymnax
import distrax
from typing import NamedTuple
from dataclasses import replace

from util.wrappers import LogWrapper, FlattenObservationWrapper
from util.networks_equinox import create_actor_critic_network

def make_HiPPO_legs(N):
    B = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = B[:, jnp.newaxis] * B[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A, B

def discretize(A, B, C, step):
    I = jnp.eye(A.shape[0])
    BL = jnp.linalg.inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C

class LinearStateSpaceLayer(eqx.Module):
    """
        State space layer initialized with the hippo matrix
    """
    A: chex.Array
    B: chex.Array
    C: chex.Array
    # D: Array # this is zero

    # log_step: float = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    cnn_mode: bool = eqx.field(static=True)
    
    def __init__(self, key, hidden_size):
        c_key, step_key = jax.random.split(key, 2)
        self.A, self.B = make_HiPPO_legs(hidden_size)
        self.B = self.B[:, jnp.newaxis]
        self.C = jax.random.normal(c_key, (1, hidden_size))
        # self.D = jnp.zeros((hidden_size, out_size))

        # Step parameter
        # self.log_step = jax.random.uniform(step_key, (1,)) * (
        #     jnp.log(0.1) - jnp.log(0.01)
        # ) + jnp.log(0.01)
        # self.log_step = jnp.log()

        self.hidden_size = hidden_size
        self.cnn_mode = False

    def __call__(self, in_x, hidden_state, done=False, **kwargs):
        # step = jnp.exp(self.log_step)
        step = 1.0 / in_x.shape[0]
        A, B, C = discretize(self.A, self.B, self.C, step=step)

        if not self.cnn_mode:
            def step_through_time(h, x):
                h_new = jnp.add(
                    A @ h,
                    B @ x
                )
                return h_new, C @ h_new 
            new_hidden_state, output = jax.lax.scan(
                step_through_time, 
                hidden_state,
                in_x[:, jnp.newaxis],
            )

        else:
            def K_conv(L):
                return jnp.array([
                    (C @ jnp.linalg.matrix_power(A, l) @ B).reshape() 
                    for l in range(L)
                ])
            
            def causal_convolution(input, K, nofft=False):
                if nofft:
                    return jnp.convolve(input, K, mode="full")[: input.shape[0]]
                else:
                    assert K.shape[0] == input.shape[0]
                    xd = jnp.fft.rfft(jnp.pad(input, (0, K.shape[0])))
                    Kd = jnp.fft.rfft(jnp.pad(K, (0, input.shape[0])))
                    out = xd * Kd
                    return jnp.fft.irfft(out)[: input.shape[0]]
            
            K = K_conv(1)
            output = causal_convolution(in_x, K, nofft=True)
            new_hidden_state = None

        return output.squeeze(), new_hidden_state

class RecurrentActorCritics(eqx.Module):
    """
        Recurrent Actor-Critic networks
    """
    shared_layers: list
    actor_layers: list
    critic_layers: list

    def __init__(self, key, in_shape, num_env_actions):
        keys = jax.random.split(key, 4)
        self.shared_layers = [
            eqx.nn.Linear(in_shape, 128, key=keys[0]),
            # eqx.nn.Linear(128, 128, key=keys[1])
            LinearStateSpaceLayer(keys[1], 128)
        ]
        self.actor_layers = [
            # eqx.nn.Linear(128, 128, key=keys[2]),
            eqx.nn.Linear(128, num_env_actions, key=keys[2])
        ]
        self.critic_layers = [
            # eqx.nn.Linear(128, 128, key=keys[2]),
            eqx.nn.Linear(128, 1, key=keys[3])
        ]

    def __call__(self, x, hidden_state, done):
        shared = jax.nn.selu(self.shared_layers[0](x))
        shared, hidden_state = self.shared_layers[1](shared, hidden_state, done)
        shared = jax.nn.selu(shared)

        actor = shared
        for layer in self.actor_layers[:-1]:
            actor = jax.nn.selu(layer(actor))
        actor_out = self.actor_layers[-1](actor)

        critic = shared
        for layer in self.critic_layers[:-1]:
            critic = jax.nn.selu(layer(critic))
        critic_out = self.critic_layers[-1](critic)

        return distrax.Categorical(logits=actor_out), critic_out.squeeze(), hidden_state

    def call_on_trace(self, x, hidden_state, done):
        shared = jax.nn.selu(jax.vmap(self.shared_layers[0])(x))
        shared, hidden_state = jax.vmap(self.shared_layers[1], in_axes=(0, None))(shared.T, hidden_state, done)
        shared = jax.nn.selu(shared.T)

        actor = shared
        for layer in self.actor_layers[:-1]:
            actor = jax.nn.selu(layer(actor))
        actor_out = jax.vmap(self.actor_layers[-1])(actor)

        critic = shared
        for layer in self.critic_layers[:-1]:
            critic = jax.nn.selu(layer(critic))
        critic_out = jax.vmap(self.critic_layers[-1])(critic)

        return distrax.Categorical(logits=actor_out), critic_out.squeeze(), hidden_state

@chex.dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 100.0
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0 # Depends on the reward scaling !
    ent_coef: float = 0.01
    vf_coef: float = 0.25

    total_timesteps: int = 5e6
    num_envs: int = 6
    num_steps: int = 64 # steps per environment
    num_minibatches: int = 4 # Number of mini-batches
    update_epochs: int = 4 # K epochs to update the policy
    # to be filled in runtime:
    batch_size: int = 0 # batch size (num_envs * num_steps)
    minibatch_size: int = 0 # mini-batch size (batch_size / num_minibatches)
    num_iterations: int = 0 # number of iterations (total_timesteps / num_steps / num_envs)

    seed: int = 4
    dpo_loss: bool = False
    debug: bool = False
    

# Define a simple tuple to hold the state of the environment. 
# This is the format we will use to store transitions in our buffer.
@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array
    log_prob: chex.Array
    info: chex.Array

class TrainState(NamedTuple):
    actor_critic: eqx.Module
    optimizer_state: optax.OptState

# Jit the returned function, not this function
def create_ppo_train_object(
        env,
        env_params,
        config_params: dict = {}
    ):

    # setup env (wrappers) and config
    env = LogWrapper(env)
    # env = FlattenObservationWrapper(env)
    observation_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    num_actions = action_space.n

    config = PPOConfig(**config_params)
    # setting runtime parameters
    num_iterations = config.total_timesteps // config.num_steps // config.num_envs
    minibatch_size = config.num_envs * config.num_steps // config.num_minibatches
    batch_size = minibatch_size * config.num_minibatches
    config = replace(
        config,
        num_iterations=num_iterations,
        minibatch_size=minibatch_size,
        batch_size=batch_size
    )

    # rng keys
    rng = jax.random.PRNGKey(config.seed)
    rng, network_key, reset_key = jax.random.split(rng, 3)

    # networks
    # actor, critic = create_actor_critic_network(
    #     key=network_key, 
    #     in_shape=observation_space.shape[0],
    #     actor_features=[64, 64], 
    #     critic_features=[64, 64], 
    #     num_env_actions=num_actions
    # )
    actor_critic = RecurrentActorCritics(network_key, observation_space.shape[0], num_actions)

    # optimizer
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.num_minibatches * config.update_epochs))
            / config.num_iterations
        )
        return config.learning_rate * frac
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(
            learning_rate=linear_schedule if config.anneal_lr else config.learning_rate,
            eps=1e-5
        ),
    )
    optimizer_state = optimizer.init(actor_critic)

    train_state = TrainState(
        actor_critic=actor_critic,
        optimizer_state=optimizer_state
    )

    rng, key = jax.random.split(rng)
    reset_key = jax.random.split(key, config.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)

    def eval_func(train_state, rng):

        def step_env(carry):
            rng, obs, env_state, done, hidden_state, episode_reward = carry
            rng, action_key, step_key = jax.random.split(rng, 3)
            action_dist, _, hidden_state = train_state.actor_critic(obs, hidden_state, done)
            action = action_dist.sample(seed=action_key)
            # action = jnp.argmax(action_dist.logits) # deterministic
            obs, env_state, reward, done, _ = env.step(step_key, env_state, action, env_params)
            episode_reward += reward
            return (rng, obs, env_state, done, hidden_state, episode_reward)
        
        def cond_func(carry):
            _, _, _, done, _, _ = carry
            return jnp.logical_not(done)
        
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)
        done = False
        episode_reward = 0.0
        hidden_state = jnp.zeros((128,))

        rng, obs, env_state, done, hidden_state, episode_reward = jax.lax.while_loop(
            cond_func, 
            step_env, 
            (rng, obs, env_state, done, hidden_state, episode_reward)
        )

        return episode_reward

    def train_func(rng=rng):
        
        # functions prepended with _ are called in jax.lax.scan of train_step

        def _env_step(runner_state, _):
            train_state, env_state, last_obs, last_done, h_state, rng = runner_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            action_dist, value, h_state = jax.vmap(train_state.actor_critic)(last_obs, h_state, last_done)
            # value = jax.vmap(train_state.critic)(last_obs)
            action, log_prob = action_dist.sample_and_log_prob(seed=sample_key)

            # take a step in the environment
            rng, key = jax.random.split(rng)
            step_key = jax.random.split(key, config.num_envs)
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_key, env_state, action, env_params)

            # Build a single transition. Jax.lax.scan will build the batch
            # returning num_steps transitions.
            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
                info=info
            )

            runner_state = (train_state, env_state, obsv, done, h_state, rng)
            return runner_state, transition
        
        def _calculate_gae(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            value, reward, done = (
                transition.value,
                transition.reward,
                transition.done,
            )
            delta = reward + config.gamma * next_value * (1 - done) - value
            gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
            return (gae, value), (gae, gae + value)
        
        def _update_epoch(update_state, _):
            """ Do one epoch of update"""

            @eqx.filter_value_and_grad(has_aux=True)
            def __ppo_los_fn(params, trajectory_minibatch, advantages, returns, hidden_state):
                action_dist, value, _ = jax.vmap(params.call_on_trace)(trajectory_minibatch.observation, hidden_state, trajectory_minibatch.done)
                log_prob = action_dist.log_prob(trajectory_minibatch.action)
                entropy = action_dist.entropy().mean()

                def ___ppo_actor_los():
                    # actor loss 
                    ratio = jnp.exp(log_prob - trajectory_minibatch.log_prob)
                    _advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    actor_loss1 = _advantages * ratio
                    actor_loss2 = (
                        jnp.clip(
                            ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                        ) * _advantages
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
                    return actor_loss
                
                def ___dpo_actor_los():
                    # dpo:
                    alpha = 2
                    beta = 0.6
                    ratio = jnp.exp(log_prob - trajectory_minibatch.log_prob)
                    norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    def when_pos(a):
                        return jax.nn.relu(
                        (ratio - 1) * a 
                        - alpha * jax.nn.tanh((ratio - 1) * a / alpha)
                    )
                    def when_neg(a):
                        return jax.nn.relu(
                        jnp.log(ratio) * a
                        - beta * jax.nn.tanh(jnp.log(ratio) * a / beta)
                    )
                    drift = jnp.where(
                        norm_advantages >= 0.0, 
                        when_pos(norm_advantages),
                        when_neg(norm_advantages)
                    )
                    actor_loss = -(ratio * norm_advantages - drift).mean()
                    return actor_loss
                
                if config.dpo_loss:
                    actor_loss = ___dpo_actor_los()
                else:
                    actor_loss = ___ppo_actor_los() 

                value_pred_clipped = trajectory_minibatch.value + (
                    jnp.clip(
                        value - trajectory_minibatch.value, -config.clip_coef_vf, config.clip_coef_vf
                    )
                )
                value_losses = jnp.square(value - returns)
                value_losses_clipped = jnp.square(value_pred_clipped - returns)
                value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                # Total loss
                total_loss = (
                    actor_loss 
                    + config.vf_coef * value_loss
                    - config.ent_coef * entropy
                )
                return total_loss, (actor_loss, value_loss, entropy)
            
            def __update_over_minibatch(train_state: TrainState, minibatch):
                trajectory_mb, advantages_mb, returns_mb, initial_hidden_state = minibatch
                (total_loss, _), grads = __ppo_los_fn(
                    train_state.actor_critic, trajectory_mb, advantages_mb, returns_mb, initial_hidden_state
                )
                updates, optimizer_state = optimizer.update(grads, train_state.optimizer_state)
                new_networks = optax.apply_updates(train_state.actor_critic, updates)
                train_state = TrainState(
                    actor_critic=new_networks,
                    optimizer_state=optimizer_state
                )
                return train_state, total_loss
            
            train_state, trajectory_batch, advantages, returns, initial_hidden_state, rng = update_state
            rng, key = jax.random.split(rng)

            batch_idx = jax.random.permutation(key, config.num_envs)
            batch = (trajectory_batch, advantages, returns)
            batch = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1), batch)
            batch = (batch[0], batch[1], batch[2], initial_hidden_state)

            # take from the batch in a new order (the order of the randomized batch_idx)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, batch_idx, axis=0), batch
            )
            # split in minibatches
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
            )

            # update over minibatches
            train_state, total_loss = jax.lax.scan(
                __update_over_minibatch, train_state, minibatches
            )
            update_state = (train_state, trajectory_batch, advantages, returns, initial_hidden_state, rng)
            return update_state, total_loss

        def train_step(runner_state, _):

            _, _, _, _, initial_hidden_state, _ = runner_state

            # Do rollout of single trajactory
            runner_state, trajectory_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # calculate gae
            train_state, env_state, last_obs, last_done, new_hidden_state, rng = runner_state
            _, last_value, _ = jax.vmap(train_state.actor_critic)(last_obs, new_hidden_state, last_done)
            _, (advantages, returns) = jax.lax.scan(
                _calculate_gae,
                (jnp.zeros_like(last_value), last_value),
                trajectory_batch,
                reverse=True,
                unroll=16
            )
    
            # Do update epochs
            update_state = (train_state, trajectory_batch, advantages, returns, initial_hidden_state, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )

            train_state = update_state[0]
            metric = trajectory_batch.info
            metric["loss_info"] = loss_info
            rng = update_state[-1]

            rng, eval_key = jax.random.split(rng)
            eval_rewards = eval_func(train_state, eval_key)
            metric["eval_rewards"] = eval_rewards

            # Debugging mode from the copied logging wrapper
            if config.debug:
                def callback(info):
                    # return_values = info["returned_episode_returns"][info["returned_episode"]]
                    # timesteps = info["timestep"][info["returned_episode"]] * config.num_envs
                    # for t in range(len(timesteps)):
                    #     print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                    print(f'timestep={(info["timestep"][-1][0] * config.num_envs)}, eval rewards={info["eval_rewards"]}')

                jax.debug.callback(callback, metric)

            

            runner_state = (train_state, env_state, last_obs, last_done, new_hidden_state, rng)
            return runner_state, metric 

        rng, key = jax.random.split(rng)
        initial_hidden_state = jnp.zeros((config.num_envs, 128))
        initial_done = jnp.zeros((config.num_envs,))
        runner_state = (train_state, env_state, obsv, initial_done, initial_hidden_state, key)
        runner_state, metrics = jax.lax.scan(
            train_step, runner_state, None, config.num_iterations
        )

        return {
            "runner_state": runner_state, 
            "metrics": metrics,
        }

    return train_func

if __name__ == "__main__":
    num_envs = 6
    env_name = "CartPole-v1"
    env, env_params = gymnax.make(env_name)
    train_func = create_ppo_train_object(
        env,
        env_params, 
        config_params = {
            "debug": True,
            "num_envs": num_envs,
        }
    )
    train_func_jit = eqx.filter_jit(train_func, backend="cpu")
    out = train_func_jit()
    info = out["metrics"]
    return_values = info["returned_episode_returns"][info["returned_episode"]]
    timesteps = info["timestep"][info["returned_episode"]] * num_envs