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
    num_steps: int = 128 # steps per environment
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
    actor: eqx.Module
    critic: eqx.Module
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
    actor, critic = create_actor_critic_network(
        key=network_key, 
        in_shape=observation_space.shape[0],
        actor_features=[64, 64], 
        critic_features=[64, 64], 
        num_env_actions=num_actions
    )

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
    optimizer_state = optimizer.init({
        "actor": actor,
        "critic": critic
    })

    train_state = TrainState(
        actor=actor,
        critic=critic,
        optimizer_state=optimizer_state
    )

    rng, key = jax.random.split(rng)
    reset_key = jax.random.split(key, config.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)

    def eval_func(train_state, rng):

        def step_env(carry):
            rng, obs, env_state, done, episode_reward = carry
            action_dist = train_state.actor(obs)
            action = jnp.argmax(action_dist.logits) # deterministic
            rng, step_key = jax.random.split(rng)
            obs, env_state, reward, done, _ = env.step(step_key, env_state, action, env_params)
            episode_reward += reward
            return (rng, obs, env_state, done, episode_reward)
        
        def cond_func(carry):
            _, _, _, done, _ = carry
            return jnp.logical_not(done)
        
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)
        done = False
        episode_reward = 0.0

        rng, obs, env_state, done, episode_reward = jax.lax.while_loop(cond_func, step_env, (rng, obs, env_state, done, episode_reward))

        return episode_reward

    def train_func(rng=rng):
        
        # functions prepended with _ are called in jax.lax.scan of train_step

        def _env_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            action_dist = jax.vmap(train_state.actor)(last_obs)
            value = jax.vmap(train_state.critic)(last_obs)
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

            runner_state = (train_state, env_state, obsv, rng)
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
            def __ppo_los_fn(params, trajectory_minibatch, advantages, returns):
                action_dist = jax.vmap(params["actor"])(trajectory_minibatch.observation)
                log_prob = action_dist.log_prob(trajectory_minibatch.action)
                entropy = action_dist.entropy().mean()
                value = jax.vmap(params["critic"])(trajectory_minibatch.observation)

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
                    return actor_loss, ratio
                
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
                    return actor_loss, ratio
                
                if config.dpo_loss:
                    actor_loss, ratio = ___dpo_actor_los()
                else:
                    actor_loss, ratio = ___ppo_actor_los() 

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
                trajectory_mb, advantages_mb, returns_mb = minibatch
                (total_loss, _), grads = __ppo_los_fn({
                        "actor": train_state.actor,
                        "critic": train_state.critic
                    }, trajectory_mb, advantages_mb, returns_mb
                )
                updates, optimizer_state = optimizer.update(grads, train_state.optimizer_state)
                new_networks = optax.apply_updates({
                    "actor": train_state.actor,
                    "critic": train_state.critic
                }, updates)
                train_state = TrainState(
                    actor=new_networks["actor"],
                    critic=new_networks["critic"],
                    optimizer_state=optimizer_state
                )
                return train_state, total_loss
            
            train_state, trajectory_batch, advantages, returns, rng = update_state
            rng, key = jax.random.split(rng)

            batch_idx = jax.random.permutation(key, config.batch_size)
            batch = (trajectory_batch, advantages, returns)
            
            # reshape (flatten over first dimension)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((config.batch_size,) + x.shape[2:]), batch
            )
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
            update_state = (train_state, trajectory_batch, advantages, returns, rng)
            return update_state, total_loss

        def train_step(runner_state, _):

            # Do rollout of single trajactory
            runner_state, trajectory_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # calculate gae
            train_state, env_state, last_obs, rng = runner_state
            last_value = jax.vmap(train_state.critic)(last_obs)
            _, (advantages, returns) = jax.lax.scan(
                _calculate_gae,
                (jnp.zeros_like(last_value), last_value),
                trajectory_batch,
                reverse=True,
                unroll=16
            )
    
            # Do update epochs
            update_state = (train_state, trajectory_batch, advantages, returns, rng)
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

            

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric 

        rng, key = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, key)
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