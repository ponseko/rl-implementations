# A clean single file implementation of PPO in Flax
# Modeled after the clearRL implementation of PPO: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
# and the implementation here: https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py
#

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import gymnasium as gym
import optax
import gymnax
import distrax
from dataclasses import replace

from flax.training.train_state import TrainState
from util.wrappers import LogWrapper, FlattenObservationWrapper
from util.networks import create_actor_critic_network

# disable jit # for debugging
# jax.config.update("jax_disable_jit", True)

@flax.struct.dataclass
class PPOParams:
    total_timesteps: int = 5e5
    """the total number of timesteps to run"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    learning_rate: float = 2.5e-4
    """the learning rate for the optimizer, disabled if linear_schedule is set"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    clip_coef_vf: float = 10.0 # Depends on the reward scaling !
    """the surrogate clipping coefficient for the value function"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.25
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    num_envs: int = 4
    """the number of environments"""
    seed: int = 0
    """the seed for the pseudo-random number generator"""
    debug: bool = False
    """Toggles printing evaluation metrics during training"""    
    dpo_loss: bool = False
    """Toggles the use of the DPO actor loss instead of the PPO loss"""

    # to be filled in runtime
    # warning, below argument rely on total_timestep, num_envs, num_steps and num_minibatches
    # Passing any of those arguments dynamically through i.e. vmap, scan, causes the code to fail
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

@flax.struct.dataclass
class AgentParams:
    shared_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict

@flax.struct.dataclass
class Transition:
    done: jnp.array
    action: jnp.array
    value: jnp.array
    reward: jnp.array
    log_prob: jnp.array
    observation: jnp.array
    info: jnp.array

# Jit the returned function, not this function
def create_ppo_train_object(env_name, config: dict = {}):
    env, env_params = gymnax.make(env_name)
    env = LogWrapper(env)
    env = FlattenObservationWrapper(env)
    observation_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    num_actions = action_space.n

    config = PPOParams(
        **config
    )

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

    shared, actor, critic = create_actor_critic_network(
        [], [64, 64], [64, 64], num_actions
    )

    # rng keys
    rng = jax.random.PRNGKey(config.seed)
    rng, shared_key, actor_key, critic_key, sample_key = jax.random.split(rng, 5)

    shared_params = shared.init(shared_key, jnp.array([observation_space.sample(sample_key)]))
    actor_params = actor.init(actor_key, shared.apply(shared_params, jnp.array([observation_space.sample(sample_key)])))
    critic_params = critic.init(critic_key, shared.apply(shared_params, jnp.array([observation_space.sample(sample_key)])))

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.num_minibatches * config.update_epochs))
            / config.num_iterations
        )
        return config.learning_rate * frac
    
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(
            learning_rate=linear_schedule if config.anneal_lr else config.learning_rate,
            eps=1e-5
        ),
    )

    agent_train_state = TrainState.create(
        apply_fn=None,
        params={
            "params": AgentParams(shared_params, actor_params, critic_params)
        },
        tx=tx
    )

    rng, key = jax.random.split(rng)
    reset_key = jax.random.split(key, config.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)

    def eval_func(train_state, rng):

        def step_env(carry):
            rng, obs, env_state, done, episode_reward = carry
            hidden = shared.apply(train_state.params["params"].shared_params, obs)
            logits = actor.apply(train_state.params["params"].actor_params, hidden)
            action = jnp.argmax(logits.logits) # deterministic
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

            # select an action
            hidden = shared.apply(train_state.params["params"].shared_params, last_obs)
            logits = actor.apply(train_state.params["params"].actor_params, hidden)
            logits_dist = logits # distrax.Categorical(logits=logits)
            rng, key = jax.random.split(rng)
            action, log_prob = logits_dist.sample_and_log_prob(seed=key)
            value = critic.apply(train_state.params["params"].critic_params, hidden)

            # take a step in the environment
            rng, key = jax.random.split(rng)
            step_key = jax.random.split(key, config.num_envs)
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_key, env_state, action, env_params)


            # Build a single transition. Jax.lax.scan will build the batch
            # returning num_steps transitions.
            transition = Transition(
                done, action, value, reward, log_prob, last_obs, info
            )

            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition
        
        def _calculate_gae(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            print(gae_and_next_value)
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
            def __ppo_los_fn(params, trajectory_minibatch, advantages, returns):
                hidden = shared.apply(params["params"].shared_params, trajectory_minibatch.observation)
                logits = actor.apply(params["params"].actor_params, hidden)
                logits_dist = logits # distrax.Categorical(logits=logits)
                log_prob = logits_dist.log_prob(trajectory_minibatch.action)
                entropy = logits_dist.entropy().mean()
                value = critic.apply(params["params"].critic_params, hidden)

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
                        return nn.relu(
                        (ratio - 1) * a 
                        - alpha * nn.tanh((ratio - 1) * a / alpha)
                    )
                    def when_neg(a):
                        return nn.relu(
                        jnp.log(ratio) * a
                        - beta * nn.tanh(jnp.log(ratio) * a / beta)
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

                # Value loss - clipped version, think this can be improved. clip_coef for value ?
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
                return total_loss, (actor_loss, value_loss, entropy)#, {"ratio_mean": ratio.mean(), "ratio_std": ratio.std()})
            
            def __update_over_minibatch(train_state: TrainState, minibatch):
                trajectory_mb, advantages_mb, returns_mb = minibatch
                grad_fn = jax.value_and_grad(__ppo_los_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, trajectory_mb, advantages_mb, returns_mb
                )
                train_state = train_state.apply_gradients(grads=grads)
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
            hidden = shared.apply(train_state.params["params"].shared_params, last_obs)
            last_value = critic.apply(train_state.params["params"].critic_params, hidden)
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
                    print(f'timestep={(info["timestep"][-1][0] * config.num_envs)[0]}, eval rewards={info["eval_rewards"]}')

                jax.debug.callback(callback, metric)

            

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric 

        rng, key = jax.random.split(rng)
        runner_state = (agent_train_state, env_state, obsv, key)
        runner_state, metrics = jax.lax.scan(
            train_step, runner_state, None, config.num_iterations
        )

        return {
            "runner_state": runner_state, 
            "metrics": metrics,
        }

    return train_func

def evaluate_agent_and_render_the_episode(agent_state: TrainState, env_name: str):
    """Evaluate the agent and render the episode"""
    env = gym.make(env_name, render_mode="human")
    obs, _ = env.reset()
    shared, actor, critic = create_actor_critic_network(
        [], [64, 64], [64, 64], env.action_space.n
    )
    done = False
    episode_reward = 0.0
    while not done:
        hidden = shared.apply(agent_state.params["params"].shared_params, obs)
        logits = actor.apply(agent_state.params["params"].actor_params, hidden)
        action = jnp.argmax(logits)
        obs, reward, done, _, _ = env.step(action.tolist())
        episode_reward += reward
        env.render()
    print(f"Episode reward: {episode_reward}")

if __name__ == "__main__":
    env_name = "CartPole-v1"
    train_func = create_ppo_train_object(
        env_name, 
        config={
            "debug": True,
            "num_envs": 6
        }
    )
    train_func_jit = jax.jit(train_func, backend="cpu")
    out = train_func_jit()
    # evaluate_agent_and_render_the_episode(
    #     out["runner_state"][0],
    #     env_name
    # )