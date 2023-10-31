# MONTE CARLO CONTROL ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given environment using the Monte Carlo algorithm

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards.   
Note that the environment is closed with a fence, so the agent cannot leave the gridworld.
### State Space:
{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
### Action Space:
There are 4 possible actions,
* 0 - Left
* 1 - Up
* 2 - Right
* 3 - Down
### Transition Probabilities
* 33.3%  - moves in the desired direction
* 66.6%  - moves in the orthogonal direction
### Rewards
* 1 - If reaches the Goal State (15)
* 0 - Otherwise
## MONTE CARLO CONTROL ALGORITHM
1. Import the necessary Libraries
2. Initialise Q-values for state-action pairs and arrays to track Q-values and the policy at each episode.
3. Initialise variables such as the number of states `nS` and actions `nA`, discount factors for different time steps `discounts`, and arrays to track learning rate and exploration rate schedules `alphas and epsilons`.
4. For each episodes 
   - Generate trajectory using `generate trajectory` function.
   -  Q-values and policies are tracked over episodes, and the best policy is extracted as the one that maximizes the Q-values for each state.
   -  For each time-step:
         - If it is first-visit Monte Carlo and it is visited, skip
         - Update the Q-values based on return.
5. The function returns the final Q-values, value function (V), and the optimal policy (pi).
## MONTE CARLO CONTROL FUNCTION
```python
from tqdm import tqdm
def mc_control (env, gamma = 1.0,
                init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,
                n_episodes = 3000, max_steps = 200, first_visit = True):
  nS, nA = env.observation_space.n, env.action_space.n
  discounts = np.logspace(0,max_steps, num=max_steps,
                          base=gamma, endpoint = False)

  alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

  epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

  pi_track = []

  Q = np.zeros((nS, nA), dtype = np.float64)
  Q_track = np.zeros((n_episodes, nS, nA), dtype = np.float64)

  select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon_decay_ratio else np.random.randint(len(Q[state]))

  for e in tqdm(range(n_episodes), leave = False):
    trajectory = generate_trajectory(select_action, Q, epsilons[e], env, max_steps)
    visited = np.zeros((nS, nA), dtype = np.bool)
    for t, (state, action, reward, _, _) in enumerate(trajectory):
      if visited[state][action] and first_visit:
        continue
      visited[state][action] = True
      n_steps = len(trajectory[t:])
      G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
      Q[state][action] += alphas[e] * (G - Q[state][action])
    Q_track[e] = Q
    pi_track.append(np.argmax(Q, axis = 1))
  V = np.max(Q, axis = 1)
  pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis = 1))}[s]
  return Q, V, pi
```
## POLICY EVALUATION:
```python
def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)
def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)
def results(env,optimal_pi,optimal_V,P):
    print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, optimal_pi, goal_state=goal_state)*100,
    mean_return(env, optimal_pi)))
goal_state = 15
results(env,optimal_pi,optimal_V,P)
```

## OUTPUT:
### Optimal Value Function
<img width="336" alt="image" src="https://github.com/Shavedha/Monte-Carlo-control/assets/93427376/4e017473-136e-43a1-95bc-72832a6fc3ea">

### State Value Function
<img width="362" alt="image" src="https://github.com/Shavedha/Monte-Carlo-control/assets/93427376/c3afda4a-c670-41b2-98fc-91cc5fd1d394">

### Action Value Function
<img width="725" alt="image" src="https://github.com/Shavedha/Monte-Carlo-control/assets/93427376/f707ce7c-2124-4c28-9c23-e09d3ffa131e">

### Success rate for optimal policy
<img width="635" alt="image" src="https://github.com/Shavedha/Monte-Carlo-control/assets/93427376/7ef0fbc3-fbcb-4a5b-92be-d4296cdc7db9">


## RESULT:
Thus a Python program is developed to find the optimal policy for the given RL environment using the Monte Carlo algorithm
