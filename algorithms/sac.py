import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
import gymnax
import flashbax
import equinox as eqx
import chex
from typing import NamedTuple

from util.wrappers import LogWrapper
from util.networks import create_actor_critic_critic_network, Q_CriticNetwork, ActorNetwork

class Alpha(eqx.Module):
    ent_coef: jnp.ndarray

    def __init__(self, ent_coef_init: float = 0.0):
        self.ent_coef = jnp.array(ent_coef_init)

    def __call__(self) -> jnp.ndarray:
        return jnp.exp(self.ent_coef)

@chex.dataclass(frozen=True)
class SacConfig:
    seed: int = 4
    num_envs: int = 4
    total_timesteps: int = 1e6
    update_every: int = 2e2
    replay_buffer_size: int = 1e4
    train_batch_size: int = 64
    alpha: float = 0.0 # is passed through an exponential function
    learn_alpha: bool = True
    target_entropy_scale_start = .5 # is linearly annealed to 0.01
    learning_rate: float = 2.5e-3
    anneal_learning_rate: bool = False
    gamma: float = 0.99
    max_grad_norm: float = 1.0
    debug: bool = False
    tau: float = 0.95
    evaluate_deterministically: bool = False

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
    q_critic1: Q_CriticNetwork
    q_critic2: Q_CriticNetwork
    actor: ActorNetwork
    alpha: Alpha
    q_critic1_target: Q_CriticNetwork
    q_critic2_target: Q_CriticNetwork
    actor_optimizer_state: optax.OptState
    q1_optimizer_state: optax.OptState
    q2_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
        
def create_train_object(
        env,
        env_params,
        config_params: dict = {}
):
    env = LogWrapper(env)
    # env = FlattenObservationWrapper(env)
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

    q1_optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(
            learning_rate=linear_schedule if config.anneal_learning_rate else config.learning_rate,
            eps=1e-5
        ),
    )

    q2_optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(
            learning_rate=linear_schedule if config.anneal_learning_rate else config.learning_rate,
            eps=1e-5
        ),
    )

    alpha_optimzer = optax.adam(
        learning_rate=config.learning_rate, eps=1e-5
    )

    # rng keys
    rng = jax.random.PRNGKey(config.seed)
    rng, network_key, sample_key, reset_key, buffer_init_step_key = jax.random.split(rng, 5)
    observation_space_shape = observation_space.sample(sample_key).shape[0]

    actor, q_critic1, q_critic2, q_critic1_target, q_critic2_target = create_actor_critic_critic_network(
        network_key, observation_space_shape, [64, 64], [64, 64], num_actions
    )
    alpha = Alpha(config.alpha)

    actor_optimizer_state = actor_optimizer.init(actor)
    q1_optimizer_state = q1_optimizer.init(q_critic1)
    q2_optimizer_state = q2_optimizer.init(q_critic2)
    alpha_optimizer_state = alpha_optimzer.init(alpha)

    train_state = TrainState(
        q_critic1=q_critic1,
        q_critic2=q_critic2,
        actor=actor,
        alpha=alpha,
        q_critic1_target=q_critic1_target,
        q_critic2_target=q_critic2_target,
        actor_optimizer_state=actor_optimizer_state,
        q1_optimizer_state=q1_optimizer_state,
        q2_optimizer_state=q2_optimizer_state,
        alpha_optimizer_state=alpha_optimizer_state
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

    def eval_func(train_state, rng):

        def step_env(carry):
            rng, obs, env_state, done, episode_reward = carry
            rng, action_key, step_key = jax.random.split(rng, 3)
            action_dist = train_state.actor(obs)
            if config.evaluate_deterministically:
                action = jnp.argmax(action_dist.logits, axis=-1)
            else:
                action = action_dist.sample(seed=action_key)
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
            @eqx.filter_value_and_grad
            def __sac_qnet_loss(params, batch):                
                q_out = jax.vmap(params)(batch["observation"])
                idx1 = jnp.arange(q_out.shape[0])
                selected_q_values = q_out[idx1, batch["action"]]
                q_loss = jnp.mean((selected_q_values - target) ** 2)
                return q_loss
            
            @eqx.filter_jit
            @eqx.filter_grad(has_aux=True)
            def __sac_actor_loss(params, batch):
                curr_action_dist = jax.vmap(params)(batch["observation"])
                curr_action_probs = curr_action_dist.probs
                curr_action_probs_log = jnp.log(curr_action_probs + 1e-8)
                q_1_outputs_curr = jax.vmap(new_critic1)(batch["observation"])
                q_2_outputs_curr = jax.vmap(new_critic2)(batch["observation"])
                q_values_curr = jnp.minimum(q_1_outputs_curr, q_2_outputs_curr)

                loss = -jnp.mean(curr_action_probs * (q_values_curr - (train_state.alpha() * curr_action_probs_log)))

                return loss, (curr_action_probs, curr_action_probs_log)
            
            @eqx.filter_jit
            @eqx.filter_grad
            def __sac_alpha_loss(params):
                def target_entropy_scale(count):
                    frac = (
                        1.0
                        - count / (config.total_timesteps // config.update_every)
                    )
                    return (config.target_entropy_scale_start * frac) + 0.01
                
                target_entropy = -(target_entropy_scale(train_state.alpha_optimizer_state[0].count)) * jnp.log(1 / num_actions)
                return -jnp.mean(
                    jnp.log(params()) 
                    * ((action_probs * action_probs_log) 
                    + target_entropy)
                )

            train_state, env_state, obsv, buffer_state, rng = runner_state
            rng, sample_key = jax.random.split(rng)
            batch = buffer.sample(buffer_state, sample_key)  # Sample
            batch = batch.experience

            # calculate q_target
            next_action_dist = jax.vmap(train_state.actor)(batch["next_observation"])
            next_action_probs = next_action_dist.probs
            next_action_probs_log = jnp.log(next_action_probs + 1e-8)
            
            q_1_target_outputs = jax.vmap(train_state.q_critic1_target)(batch["next_observation"])
            q_2_target_outputs = jax.vmap(train_state.q_critic2_target)(batch["next_observation"])

            target = (next_action_probs * (jnp.minimum(q_1_target_outputs, q_2_target_outputs) - train_state.alpha() * next_action_probs_log)).sum(axis=-1)
            target = batch["reward"] + ~batch["done"] * config.gamma * target

            # Updating Q networks
            q_1_loss, q_1_grads = __sac_qnet_loss(params=train_state.q_critic1, batch=batch)
            q_2_loss, q_2_grads = __sac_qnet_loss(params=train_state.q_critic2, batch=batch)
            updates_q1, q1_optimizer_state = q1_optimizer.update(q_1_grads, train_state.q1_optimizer_state)
            updates_q2, q2_optimizer_state = q2_optimizer.update(q_2_grads, train_state.q2_optimizer_state)
            new_critic1 = optax.apply_updates(train_state.q_critic1, updates_q1)
            new_critic2 = optax.apply_updates(train_state.q_critic2, updates_q2)

            # update target policy
            new_q_critic1_target = jax.tree.map(lambda x, y: config.tau * x + (1 - config.tau) * y, train_state.q_critic1_target, new_critic1)
            new_q_critic2_target = jax.tree.map(lambda x, y: config.tau * x + (1 - config.tau) * y, train_state.q_critic2_target, new_critic2)

            # Updating the actor
            actor_grads, (action_probs, action_probs_log) = __sac_actor_loss(params=train_state.actor, batch=batch)
            updates, actor_optimizer_state = actor_optimizer.update(actor_grads, train_state.actor_optimizer_state)
            new_actor = optax.apply_updates(train_state.actor, updates)

            if config.learn_alpha:
                # Updating alpha
                alpha_grads = __sac_alpha_loss(params=train_state.alpha)
                updates, alpha_optimizer_state = alpha_optimzer.update(alpha_grads, train_state.alpha_optimizer_state)
                new_alpha = optax.apply_updates(train_state.alpha, updates)
            else:
                new_alpha = train_state.alpha
                alpha_optimizer_state = train_state.alpha_optimizer_state

            # replace the named tuple with the updated values
            train_state = TrainState(
                q_critic1=new_critic1,
                q_critic2=new_critic2,
                actor=new_actor,
                alpha=new_alpha,
                q_critic1_target=new_q_critic1_target,
                q_critic2_target=new_q_critic2_target,
                actor_optimizer_state=actor_optimizer_state,
                q1_optimizer_state=q1_optimizer_state,
                q2_optimizer_state=q2_optimizer_state,
                alpha_optimizer_state=alpha_optimizer_state
            )

            runner_state = (train_state, env_state, obsv, buffer_state, rng)

            return runner_state, (q_1_loss, q_2_loss)

        def train_step(runner_state, _):

            # Do rollout of single trajactory
            runner_state, metric = lax.scan(
                _env_step, runner_state, None, config.update_every // config.num_envs
            )
            runner_state, (q_1_loss, q_2_loss) = _update_step(runner_state)
            rng = runner_state[-1]

            rng, eval_key = jax.random.split(rng)
            eval_rewards = eval_func(runner_state[0], eval_key)
            metric["eval_rewards"] = eval_rewards
            metric["q_1_loss"] = q_1_loss
            metric["q_2_loss"] = q_2_loss

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
