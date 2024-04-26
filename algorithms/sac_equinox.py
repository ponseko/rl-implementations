import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
import gymnax
from dataclasses import dataclass
import flashbax
import equinox as eqx
import chex
from typing import NamedTuple

from util.wrappers import LogWrapper, FlattenObservationWrapper
from util.networks_equinox import create_actor_critic_critic_network, Q_CriticNetwork, ActorNetwork

@dataclass
class SacConfig:
    seed: int = 0
    num_envs: int = 4
    total_timesteps: int = 2e6
    update_every: int = 1e3
    replay_buffer_size: int = 1e4
    alpha: float = 0.2
    learning_rate: float = 2.5e-3
    anneal_learning_rate: bool = True
    gamma: float = 0.998
    max_grad_norm: float = 0.5
    debug: bool = False

# Define a simple tuple to hold the state of the environment. This is the format we will use to store transitions in our buffer.
@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    next_observation: chex.Array
    done: chex.Array

# ToDo use this
class TrainState(NamedTuple):
    q_critic1: Q_CriticNetwork
    q_critic2: Q_CriticNetwork
    actor: ActorNetwork
    q_critic1_target: Q_CriticNetwork
    q_critic2_target: Q_CriticNetwork
    actor_optimizer_state: optax.OptState
    q_optimizer_state: optax.OptState

def create_train_object(
        env,
        env_params,
        config_params: dict = {}
):
    env = LogWrapper(env)
    env = FlattenObservationWrapper(env)
    observation_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    num_actions = action_space.n

    config = SacConfig(
        **config_params
    )

    def linear_schedule(count):
        frac = (
            1.0
            - count / (config.total_timesteps // config.update_every)
        )
        return config.learning_rate * frac
    
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(
            learning_rate=linear_schedule if config.anneal_learning_rate else config.learning_rate,
            eps=1e-5
        ),
    )

    q_optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(
            learning_rate=linear_schedule if config.anneal_learning_rate else config.learning_rate,
            eps=1e-5
        ),
    )

    # rng keys
    rng = jax.random.PRNGKey(config.seed)
    rng, network_key, sample_key, reset_key, buffer_init_step_key = jax.random.split(rng, 5)
    observation_space_shape = observation_space.sample(sample_key).shape[0]

    actor, q_critic1, q_critic2, q_critic1_target, q_critic2_target = create_actor_critic_critic_network(
        network_key, observation_space_shape, [64, 64], [64, 64], num_actions
    )

    actor_optimizer_state = actor_optimizer.init(actor)
    q_optimizer_state = q_optimizer.init({
        "q_critic1": q_critic1,
        "q_critic2": q_critic2
    })

    train_state = TrainState(
        q_critic1=q_critic1,
        q_critic2=q_critic2,
        actor=actor,
        q_critic1_target=q_critic1_target,
        q_critic2_target=q_critic2_target,
        actor_optimizer_state=actor_optimizer_state,
        q_optimizer_state=q_optimizer_state
    )

    # agent = {
    #     "q_critic1": q_critic1,
    #     "q_critic2": q_critic2,
    #     "actor": actor,
    #     "q_critic1_target": q_critic1_target,
    #     "q_critic2_target": q_critic2_target
    # }
    
    # setup buffer
    obs, dummy_env_state = env.reset(reset_key, env_params)
    buffer = flashbax.make_flat_buffer(
        max_length=int(config.replay_buffer_size), 
        min_length=int(config.update_every), 
        sample_batch_size=int(config.update_every),
        add_sequences=False,
        add_batch_size=config.num_envs if config.num_envs > 1 else None,
    )
    dummy_action = action_space.sample(sample_key)
    obs_dummy, _, reward_dummy, done_dummy, _ = env.step(buffer_init_step_key, dummy_env_state, dummy_action, env_params)
    dummy_transition = Transition(
        observation=obs_dummy,
        action=dummy_action,
        reward=reward_dummy,
        next_observation=obs_dummy,
        done=done_dummy
    )
    buffer_state = buffer.init(dummy_transition)

    reset_key = jax.random.split(reset_key, config.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
        

    def eval_func(train_state, rng):

        def step_env(carry):
            rng, obs, env_state, done, episode_reward = carry
            action_dist = train_state.actor(obs)
            action = jnp.argmax(action_dist._logits) # deterministic
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

        rng, obs, env_state, done, episode_reward = lax.while_loop(cond_func, step_env, (rng, obs, env_state, done, episode_reward))

        return episode_reward

    def train_func(rng=rng):

        # functions prepended with _ are called in lax.scan of train_step

        def _env_step(runner_state, _):
            train_state, env_state, obsv, buffer_state, rng = runner_state

            # get action
            rng, action_key = jax.random.split(rng)
            action_dist = jax.vmap(train_state.actor)(obsv)
            action = action_dist.sample(seed=action_key)

            # take step
            rng, key = jax.random.split(rng)
            step_key = jax.random.split(key, config.num_envs)
            next_obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_key, env_state, action, env_params)

            transition = Transition(
                observation=obsv,
                action=action,
                reward=reward,
                next_observation=next_obsv,
                done=done
            )

            # add to buffer
            buffer_state = buffer.add(buffer_state, transition)

            return (train_state, env_state, next_obsv, buffer_state, rng), info
        
        def _update_step(runner_state):
            
            @eqx.filter_jit
            @eqx.filter_grad
            def __sac_qnet_loss(params, batch):

                q1_out = jax.vmap(params["q_critic1"])(batch["observation"])
                q2_out = jax.vmap(params["q_critic1"])(batch["observation"])
                selected_q1_values = q1_out[index_array, next_action]
                selected_q2_values = q2_out[index_array, next_action]
                q1_loss = jnp.mean((selected_q1_values - target) ** 2)
                q2_loss = jnp.mean((selected_q2_values - target) ** 2)

                return q1_loss + q2_loss
            
            @eqx.filter_jit
            @eqx.filter_grad
            def __sac_actor_loss(params, batch):
                
                curr_action_dist = jax.vmap(params)(batch["observation"])
                curr_action, curr_log_prob = curr_action_dist.sample_and_log_prob(seed=sample_key)
                q_1_outputs_curr = jax.vmap(new_critic1)(batch["observation"])
                q_2_outputs_curr = jax.vmap(new_critic2)(batch["observation"])
                # index into the array according to next_action
                index_array = jnp.arange(q_1_outputs.shape[0])
                selected_q1_values_curr = q_1_outputs_curr[index_array, curr_action]
                selected_q2_values_curr = q_2_outputs_curr[index_array, curr_action]
                q_values_curr = jnp.minimum(selected_q1_values_curr, selected_q2_values_curr)

                loss = -jnp.mean(q_values_curr - config.alpha * curr_log_prob)

                return loss              

            train_state, env_state, obsv, buffer_state, rng = runner_state
            rng, sample_key = jax.random.split(rng)
            batch = buffer.sample(buffer_state, sample_key)  # Sample
            batch = batch.experience.first
            

            next_action_dist = jax.vmap(train_state.actor)(batch["next_observation"])
            next_action, next_log_prob = next_action_dist.sample_and_log_prob(seed=sample_key)
            q_1_outputs = jax.vmap(train_state.q_critic1_target)(batch["next_observation"])
            q_2_outputs = jax.vmap(train_state.q_critic2_target)(batch["next_observation"])
        	
            # index into the array according to next_action
            index_array = jnp.arange(q_1_outputs.shape[0])
            selected_q1_values = q_1_outputs[index_array, next_action]
            selected_q2_values = q_2_outputs[index_array, next_action]

            target = jnp.minimum(selected_q1_values, selected_q2_values) - config.alpha * next_log_prob
            target = batch["reward"] + (1.0 - batch["done"]) * config.gamma * target

            q_grads = __sac_qnet_loss(params={
                "q_critic1": train_state.q_critic1,
                "q_critic2": train_state.q_critic2
            }, batch=batch)
            updates, q_optimizer_state = q_optimizer.update(q_grads, train_state.q_optimizer_state)
            critics = optax.apply_updates({
                "q_critic1": train_state.q_critic1,
                "q_critic2": train_state.q_critic2
            }, updates)
            new_critic1 = critics["q_critic1"]
            new_critic2 = critics["q_critic2"]

            actor_grads = __sac_actor_loss(train_state.actor, batch)
            updates, actor_optimizer_state = actor_optimizer.update(actor_grads, train_state.actor_optimizer_state)
            new_actor = optax.apply_updates(train_state.actor, updates)
            
            # update target policy
            tau = 0.005
            new_q_critic1_target = jax.tree.map(lambda x, y: tau * x + (1 - tau) * y, train_state.q_critic1_target, new_critic1)
            new_q_critic2_target = jax.tree.map(lambda x, y: tau * x + (1 - tau) * y, train_state.q_critic2_target, new_critic2)

            # replace the named tuple with the updated values
            train_state = TrainState(
                q_critic1=new_critic1,
                q_critic2=new_critic2,
                actor=new_actor,
                q_critic1_target=new_q_critic1_target,
                q_critic2_target=new_q_critic2_target,
                actor_optimizer_state=actor_optimizer_state,
                q_optimizer_state=q_optimizer_state
            )

            runner_state = (train_state, env_state, obsv, buffer_state, rng)

            return runner_state

        def train_step(runner_state, _):

            # Do rollout of single trajactory
            runner_state, metric = lax.scan(
                _env_step, runner_state, None, config.update_every // config.num_envs
            )

            runner_state = _update_step(runner_state)

            rng = runner_state[-1]

            rng, eval_key = jax.random.split(rng)
            eval_rewards = eval_func(train_state, eval_key)
            metric["eval_rewards"] = eval_rewards

            # Debugging mode from the copied logging wrapper
            if config.debug:
                def callback(info):
                    print(f'timestep={(info["timestep"][-1][0] * config.num_envs)}, eval rewards={info["eval_rewards"]}')
                jax.debug.callback(callback, metric)
            # if config.debug:
            #     def callback(info):
            #         return_values = info["returned_episode_returns"][info["returned_episode"]]
            #         timesteps = info["timestep"][info["returned_episode"]] * config.num_envs
            #         for t in range(len(timesteps)):
            #             print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
            #     jax.debug.callback(callback, metric)

            

            # runner_state = (agent, env_state, last_obs, rng)
            return runner_state, metric 
        
        rng, train_key = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, buffer_state, train_key)
        runner_state, metrics = lax.scan(
            train_step, runner_state, None, config.total_timesteps // config.update_every
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
    train_func = create_train_object(
        env,
        env_params, 
        config_params = {
            "debug": True,
            "num_envs": num_envs,
        }
    )
    train_func_jit = eqx.filter_jit(train_func, backend="cpu")
    out = train_func_jit()
    # evaluate_agent_and_render_the_episode(
    #     out["runner_state"][0],
    #     env_name
    # )
    info = out["metrics"]
    return_values = info["returned_episode_returns"][info["returned_episode"]]
    timesteps = info["timestep"][info["returned_episode"]] * num_envs
    import matplotlib.pyplot as plt
    print(return_values.mean())
    plt.plot(return_values)
    plt.show()
