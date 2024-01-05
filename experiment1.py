import numpy as np
import torch

""" Trial generation functions """

def gen_baseline_trials(params, n_trials=20):
    state_reps = torch.eye(params['state_d'])
    visited_states, rewards = [], []
    for trial_idx in range(n_trials):
        if np.random.random()<.5:
            visited_states.extend([1,3,5])
            rewards.extend([0,0,10])
        else:
            visited_states.extend([2,4,6])
            rewards.extend([0,0,1])
    visited_states = state_reps[visited_states]
    rewards = torch.tensor(rewards,dtype=torch.float)
    return visited_states, rewards


def gen_reward_revaluation_trials(params, n_trials=20):
    state_reps = torch.eye(params['state_d'])
    visited_states, rewards = [], []
    for trial_idx in range(n_trials):
        if np.random.random()<.5:
            visited_states.extend([3,5])
            rewards.extend([0,1])
        else:
            visited_states.extend([4,6])
            rewards.extend([0,10])
    visited_states = state_reps[visited_states]
    rewards = torch.tensor(rewards,dtype=torch.float)
    return visited_states, rewards


def gen_transition_revaluation_trials(params, n_trials=20):
    state_reps = torch.eye(params['state_d'])
    visited_states, rewards = [], []
    for trial_idx in range(n_trials):
        if np.random.random()<.5:
            visited_states.extend([3,6])
            rewards.extend([0,1])
        else:
            visited_states.extend([4,5])
            rewards.extend([0,10])
    visited_states = state_reps[visited_states]
    rewards = torch.tensor(rewards,dtype=torch.float)
    return visited_states, rewards


def gen_control_trials(params, n_trials=20):
    state_reps = torch.eye(params['state_d'])
    visited_states, rewards = [], []
    for trial_idx in range(n_trials):
        if np.random.random()<.5:
            visited_states.extend([3,5])
            rewards.extend([0,10])
        else:
            visited_states.extend([4,6])
            rewards.extend([0,1])
    visited_states = state_reps[visited_states]
    rewards = torch.tensor(rewards,dtype=torch.float)
    return visited_states, rewards


""" Memory sampling functions. Replaced with EMModule in later experiments for added complexity """


def norm(key):
    return key/key.norm(dim=-1,keepdim=True)


def match(key, memories):
    return (norm(memories)*norm(key)).sum(-1)


def torch_append(tensor, value):
    if tensor is None:
        tensor = value.detach().clone().unsqueeze(0)
    else:
        tensor = torch.cat([tensor,value.detach().clone().unsqueeze(0)],axis=0)
    return tensor


def update_context(old_context, new_state, new_context, params):
    self_excitation, input_weight, retrieved_context_weight = params['self_excitation'], params['input_weight'], params['retrieved_context_weight']
    return self_excitation*old_context + input_weight*new_state + retrieved_context_weight*new_context
    

def gen_memories(visited_states, rewards, params):
    context_rep = torch.zeros((params['context_d'],),dtype=torch.float)+.01
    time_code = torch.zeros((params['time_d'],),dtype=torch.float)+.01
    state_memories, context_memories, time_memories, reward_memories = None, None, None, None
    time_noise = params['time_noise']
    for t in range(len(visited_states)):
        if t > 0:
            se = params['self_excitation']
            iw = params['input_weight']
            rw = params['retrieved_context_weight']
            context_rep = se/(se+iw)*context_rep + iw/(se+iw)*visited_states[t-1]
            _, retrieved_context, _, _ = sample_memory((state_memories, context_memories, time_memories, reward_memories),
                                                       (visited_states[t], context_rep, time_code, rewards[t]), params, mode='softmax')
            context_rep = (se+iw)*context_rep + rw*retrieved_context
            time_code += torch.randn_like(time_code)*time_noise
 
        state_memories = torch_append(state_memories, visited_states[t])
        context_memories = torch_append(context_memories, context_rep)
        time_memories = torch_append(time_memories, time_code)
        reward_memories = torch_append(reward_memories, rewards[t])
        
    return state_memories, context_memories, time_memories, reward_memories


def sample_memory(memories, query, params, mode='sample', return_idx=False):
    state_memories, context_memories, time_memories, reward_memories = memories
    state_weight, context_weight, time_weight = params['state_weight'], params['context_weight'], params['time_weight']
    temperature = params['temperature']
    state, context, time, _ = query

    state_match = match(state,state_memories)
    context_match = match(context,context_memories)
    time_match = match(time,time_memories)
    total_match = ((state_weight*state_match + context_weight*context_match + time_weight*time_match)/temperature).softmax(-1)

    if mode=='sample':
        index = torch.multinomial(total_match,1).item()
        if return_idx:
            return state_memories[index], context_memories[index], time_memories[index], reward_memories[index], index
        else:
            return state_memories[index], context_memories[index], time_memories[index], reward_memories[index], index
    elif mode=='argmax':
        index = total_match.argmax().item()
        if return_idx:
            return state_memories[index], context_memories[index], time_memories[index], reward_memories[index], index
        else:
            return state_memories[index], context_memories[index], time_memories[index], reward_memories[index], index
    elif mode=='softmax':
        return (total_match.unsqueeze(-1)*state_memories).sum(0), (total_match.unsqueeze(-1)*context_memories).sum(0), (total_match.unsqueeze(-1)*time_memories).sum(0), (total_match*reward_memories).sum(0)
    else:
        raise NotImplementedError(f'Mode {mode} not implemented. Try one of ["sample", "argmax", "softmax"].')


def sample_memory_sequential(memories, starting_query, params):
    state_memories, context_memories, time_memories, reward_memories = memories
    starting_state, starting_context, starting_time, _ = starting_query
    n_simulations, n_steps = params['n_simulations'], params['n_steps']
    retrieved_states = np.zeros((n_simulations, n_steps, params['state_d']))
    retrieved_contexts = np.zeros((n_simulations, n_steps, params['context_d']))
    retrieved_times = np.zeros((n_simulations, n_steps, params['time_d']))
    retrieved_rewards = np.zeros((n_simulations, n_steps))
    retrieved_memory_idxs = np.zeros((n_simulations, n_steps),dtype=int)
    for sim_idx in range(n_simulations):
        state = starting_state
        context = starting_context
        time = starting_time
        params_new = {k:v for k,v in params.items()}
        for step_idx in range(n_steps):
            retrieved_state, retrieved_context, retrieved_time, retrieved_reward, retrieved_memory_idx = sample_memory((state_memories, context_memories, time_memories, reward_memories),
                                                                                                (state, context, time, 0), params_new, mode='sample', return_idx=True)
            context = update_context(context, retrieved_state, retrieved_context, params)
            retrieved_states[sim_idx,step_idx] = retrieved_state.detach().clone().numpy()
            retrieved_contexts[sim_idx,step_idx] = retrieved_context.detach().clone().numpy()
            retrieved_times[sim_idx,step_idx] = retrieved_time.detach().clone().numpy()
            retrieved_rewards[sim_idx,step_idx] = retrieved_reward.item()
            retrieved_memory_idxs[sim_idx,step_idx] = retrieved_memory_idx
            params_new['state_weight'] = 0
            try:
                params_new['context_weight'] /= params_new['context_weight']+params_new['time_weight']
                params_new['time_weight'] /= params_new['context_weight']+params_new['time_weight']
            except ZeroDivisionError:
                params_new['context_weight'] = 0
                params_new['time_weight'] = 0
    return retrieved_states, retrieved_contexts, retrieved_times, retrieved_rewards, retrieved_memory_idxs


""" Task performance functions """


def estimate_reward_from_starting_state(memories, starting_state, params, return_trajectories=False):
    starting_context = memories[1][-1]
    starting_time = memories[2][-1]
    starting_query = (starting_state,starting_context, starting_time, None)
    sampled_trajectories = sample_memory_sequential(memories, starting_query, params)
    estimated_reward = sampled_trajectories[3].sum(axis=-1).mean() # Sum over steps in each sim and avg over sims
    if return_trajectories:
        return estimated_reward, sampled_trajectories
    else:
        return estimated_reward


def run_experiment(params):
    revaluation_scores = np.zeros((params['n_participants'],3))
    for participant_idx in range(params['n_participants']):
        visited_states_baseline, rewards_baseline = gen_baseline_trials(params)
        memories = gen_memories(visited_states_baseline, rewards_baseline, params)
        estimated_reward_state_one_baseline = estimate_reward_from_starting_state(memories,torch.eye(7)[1],params)
        estimated_reward_state_two_baseline = estimate_reward_from_starting_state(memories,torch.eye(7)[2],params)

        visited_states_reward_reval, rewards_reward_reval = gen_reward_revaluation_trials(params)
        visited_states_reward_reval = torch.cat([visited_states_baseline,visited_states_reward_reval],axis=0)
        rewards_reward_reval = torch.cat([rewards_baseline,rewards_reward_reval],axis=0)
        memories = gen_memories(visited_states_reward_reval, rewards_reward_reval, params)
        estimated_reward_state_one_reward_reval = estimate_reward_from_starting_state(memories,torch.eye(7)[1],params)
        estimated_reward_state_two_reward_reval = estimate_reward_from_starting_state(memories,torch.eye(7)[2],params)

        visited_states_transition_reval, rewards_transition_reval = gen_transition_revaluation_trials(params)
        visited_states_transition_reval = torch.cat([visited_states_baseline,visited_states_transition_reval],axis=0)
        rewards_transition_reval = torch.cat([rewards_baseline,rewards_transition_reval],axis=0)
        memories = gen_memories(visited_states_transition_reval, rewards_transition_reval, params)
        estimated_reward_state_one_transition_reval = estimate_reward_from_starting_state(memories,torch.eye(7)[1],params)
        estimated_reward_state_two_transition_reval = estimate_reward_from_starting_state(memories,torch.eye(7)[2],params)

        visited_states_control, rewards_control = gen_control_trials(params)
        visited_states_control = torch.cat([visited_states_baseline,visited_states_control],axis=0)
        rewards_control = torch.cat([rewards_baseline,rewards_control],axis=0)
        memories = gen_memories(visited_states_control, rewards_control, params)
        estimated_reward_state_one_control = estimate_reward_from_starting_state(memories,torch.eye(7)[1],params)
        estimated_reward_state_two_control = estimate_reward_from_starting_state(memories,torch.eye(7)[2],params)

        state_one_preference_baseline = estimated_reward_state_one_baseline-estimated_reward_state_two_baseline
        state_one_preference_reward_reval = estimated_reward_state_one_reward_reval-estimated_reward_state_two_reward_reval
        state_one_preference_transition_reval = estimated_reward_state_one_transition_reval-estimated_reward_state_two_transition_reval
        state_one_preference_control = estimated_reward_state_one_control-estimated_reward_state_two_control

        revaluation_scores[participant_idx,0] = state_one_preference_baseline-state_one_preference_reward_reval
        revaluation_scores[participant_idx,1] = state_one_preference_baseline-state_one_preference_transition_reval
        revaluation_scores[participant_idx,2] = state_one_preference_baseline-state_one_preference_control
    return revaluation_scores