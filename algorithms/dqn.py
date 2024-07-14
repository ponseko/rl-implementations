import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
import gymnax
import flashbax
import equinox as eqx
import chex
from typing import NamedTuple, List

from util.wrappers import LogWrapper
from util.networks import Q_CriticNetwork 

def create_dqn_network(
        key: chex.PRNGKey,
        in_shape: int,
        actor_features: List[int],
        critic_features: List[int],
        num_env_actions: int,
):
    critic = Q_CriticNetwork(
        key, in_shape, critic_features, num_env_actions
    )
    target_critic = jax.tree_util.tree_map(lambda x: x, critic)
    return critic, target_critic


@chex.dataclass(frozen=True)
class DqnConfig:
    seed: int = 4
    num_envs: int = 4
    total_timesteps: int = 1e6
    update_every: int = 2e2
    replay_buffer_size: int = 1e4
    train_batch_size: int = 64
    epsilon: float = 0.3
    learning_rate: float = 2.5e-3
    anneal_learning_rate: bool = False
    gamma: float = 0.99
    max_grad_norm: float = 1.0
    tau: float = 0.95
    debug: bool = False

# Define a simple tuple to hold the state of the environment. 
# This is the format we will use to store transitions in our buffer.
@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    next_observation: chex.Array
    done: chex.Array

# ToDo use this
class TrainState(NamedTuple):
    critic: Q_CriticNetwork
    critic_target: Q_CriticNetwork
    critic_optimizer_state: optax.OptState
        
def create_train_object(
        env,
        env_params,
        config_params: dict = {}
):
    env = LogWrapper(env)
    observation_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    num_actions = action_space.n

    config = DqnConfig(
        **config_params
    )

    def linear_schedule(count):
        frac = (
            1.0
            - count / (config.total_timesteps // config.update_every)
        )
        return config.learning_rate * frac
    
    critic_optimizer = optax.chain(
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

    critic, critic_target = create_dqn_network(
        network_key, observation_space_shape, [64, 64], [64, 64], num_actions
    )
    actor_optimizer_state = critic_optimizer.init(critic)
    train_state = TrainState(
        critic=critic,
        critic_optimizer_state=actor_optimizer_state,
        critic_target=critic_target
    )

    # setup buffer
    obs, dummy_env_state = env.reset(reset_key, env_params)
    buffer = flashbax.make_item_buffer(
        max_length=int(config.replay_buffer_size), 
        min_length=int(config.train_batch_size), 
        sample_batch_size=int(config.train_batch_size),
        add_sequences=False,
        add_batches=config.num_envs if config.num_envs > 1 else None,
    )
    dummy_action = action_space.sample(sample_key)
    obs_dummy, _, reward_dummy, done_dummy, dummy_info = env.step(buffer_init_step_key, dummy_env_state, dummy_action, env_params)
    dummy_transition = Transition(
        observation=obs_dummy,
        action=dummy_action,
        reward=reward_dummy,
        next_observation=obs_dummy,
        done=done_dummy,
    )
    buffer_state = buffer.init(dummy_transition)

    reset_key = jax.random.split(reset_key, config.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)

    def eval_func(train_state: TrainState, rng):

        def step_env(carry):
            rng, obs, env_state, done, episode_reward = carry
            rng, step_key = jax.random.split(rng)
            q_out = train_state.critic(obs)
            action = jnp.argmax(q_out, axis=-1)
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
            rng, sample_key, epsilon_key, step_key = jax.random.split(rng, 4)
            sample_keys = jax.random.split(sample_key, config.num_envs)
            epsilon_keys = jax.random.split(epsilon_key, config.num_envs)
            q_out = jax.vmap(train_state.critic)(obsv)
            random_actions = jax.vmap(action_space.sample)(sample_keys)
            greedy_actions = jnp.argmax(q_out, axis=-1)
            action = jnp.where(
                jax.vmap(jax.random.bernoulli, in_axes=(0, None))(epsilon_keys, config.epsilon),
                random_actions,
                greedy_actions
            )

            # take step
            step_key = jax.random.split(step_key, config.num_envs)
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

            @eqx.filter_value_and_grad
            def __dqn_critic_loss(params, batch):                
                q_out_1 = jax.vmap(params)(batch["observation"])
                idx1 = jnp.arange(q_out_1.shape[0])
                selected_q_values = q_out_1[idx1, batch["action"]]
                q_loss = jnp.mean((selected_q_values - target) ** 2)
                return q_loss

            train_state, env_state, obsv, buffer_state, rng = runner_state
            rng, sample_key = jax.random.split(rng)
            batch = buffer.sample(buffer_state, sample_key)  # Sample
            batch = batch.experience

            # calculate q_target

            q_target_output = jax.vmap(train_state.critic_target)(batch["next_observation"])
            target = batch["reward"] + ~batch["done"] * config.gamma * jnp.max(q_target_output, axis=-1)

            # Updating Q networks
            critic_loss, critic_grads = __dqn_critic_loss(train_state.critic, batch)
            updates_critic, critic_optimizer_state = critic_optimizer.update(critic_grads, train_state.critic_optimizer_state)
            new_critic = optax.apply_updates(train_state.critic, updates_critic)

            # update target policy
            new_critic_target = jax.tree.map(lambda x, y: config.tau * x + (1 - config.tau) * y, train_state.critic_target, new_critic)

            # replace the named tuple with the updated values
            train_state = TrainState(
                critic=new_critic,
                critic_optimizer_state=critic_optimizer_state,
                critic_target=new_critic_target
            )

            runner_state = (train_state, env_state, obsv, buffer_state, rng)

            return runner_state, critic_loss

        def train_step(runner_state, _):

            # Do rollout of single trajactory
            runner_state, metric = lax.scan(
                _env_step, runner_state, None, config.update_every // config.num_envs
            )
            runner_state, critic_loss = _update_step(runner_state)
            rng = runner_state[-1]

            rng, eval_key = jax.random.split(rng)
            eval_rewards = eval_func(runner_state[0], eval_key)
            metric["eval_rewards"] = eval_rewards
            metric["critic_loss"] = critic_loss

            if config.debug:
                def callback(info):
                    print(f'timestep={(info["timestep"][-1][0] * config.num_envs)}, eval rewards={info["eval_rewards"]}')
                jax.debug.callback(callback, metric)

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
