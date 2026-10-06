"""
=============================================================================
  Actor-Critic Methods — Step-by-Step Python Simulations
=============================================================================

Run:  python actor_critic_simulations.py

Every internal variable is printed so you can trace exactly how
the actor (policy) and critic (value function) learn together.

Simulations:
  1. One-Step Actor-Critic (A2C core loop)
  2. Actor-Critic with Eligibility Traces
  3. Advantage Actor-Critic (A2C) on CartPole-style env
  4. REINFORCE baseline comparison (why critic helps)
  5. Continuous Action Space Actor-Critic (Gaussian policy)
  6. Entropy-Regularised Actor-Critic
=============================================================================
"""

import math
import random
import numpy as np

random.seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def softmax(logits):
    """Numerically stable softmax: π(a|s) = exp(logit_a) / Σ exp(logit_j)"""
    logits = np.array(logits, dtype=float)
    logits -= logits.max()                    # stability trick
    exp_l = np.exp(logits)
    return exp_l / exp_l.sum()


def sample_action(probs):
    """Sample action from probability distribution."""
    return np.random.choice(len(probs), p=probs)


def print_header(title):
    w = 72
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT: Simple Grid World
# ─────────────────────────────────────────────────────────────────────────────
# 1D chain:  [0] - [1] - [2] - ... - [N-1]
#            left-terminal              right-terminal (+1 reward)
# Actions: 0=left, 1=right
# Reward: +1 when reaching state N-1, −1 when reaching state 0, else 0

class ChainWalk:
    """1D random walk environment for clear demonstration."""
    def __init__(self, n_states=7):
        self.n_states = n_states
        self.state = n_states // 2

    def reset(self):
        self.state = self.n_states // 2
        return self.state

    def step(self, action):
        """action: 0=left, 1=right"""
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.n_states - 1, self.state + 1)

        if self.state == self.n_states - 1:
            return self.state, +1.0, True
        elif self.state == 0:
            return self.state, -1.0, True
        else:
            return self.state, 0.0, False

    def get_features(self, state):
        """One-hot encoding of state."""
        x = np.zeros(self.n_states)
        x[state] = 1.0
        return x


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT: CartPole-like (simplified)
# ─────────────────────────────────────────────────────────────────────────────
# State: [position, velocity, angle, angular_velocity]
# Actions: 0=push-left, 1=push-right
# Done when |angle| > 0.25 or |position| > 2.5 or 200 steps

class SimpleCartPole:
    """Simplified cart-pole for demonstration (no external deps needed)."""
    def __init__(self):
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 0.5
        self.force_mag = 10.0
        self.dt = 0.02
        self.state = None

    def reset(self):
        self.state = np.array([
            random.uniform(-0.05, 0.05),   # position
            random.uniform(-0.05, 0.05),   # velocity
            random.uniform(-0.05, 0.05),   # angle
            random.uniform(-0.05, 0.05),   # angular velocity
        ])
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        total_mass = self.cart_mass + self.pole_mass
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Physics (simplified Euler integration)
        temp = (force + self.pole_mass * self.pole_length * theta_dot**2 * sin_t) / total_mass
        theta_acc = (self.gravity * sin_t - cos_t * temp) / (
            self.pole_length * (4.0/3.0 - self.pole_mass * cos_t**2 / total_mass))
        x_acc = temp - self.pole_mass * self.pole_length * theta_acc * cos_t / total_mass

        x += self.dt * x_dot
        x_dot += self.dt * x_acc
        theta += self.dt * theta_dot
        theta_dot += self.dt * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        done = abs(theta) > 0.25 or abs(x) > 2.5 or self.steps >= 200
        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done

    def get_features(self, state):
        """Normalised state features + bias."""
        return np.array([
            state[0] / 2.5,       # normalised position
            state[1] / 3.0,       # normalised velocity
            state[2] / 0.25,      # normalised angle
            state[3] / 3.0,       # normalised angular velocity
            1.0,                  # bias term
        ])


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION 1: One-Step Actor-Critic
# ─────────────────────────────────────────────────────────────────────────────
#
# The core idea:
#
#   CRITIC:  learns V(s) to evaluate how good a state is
#            Update:  w ← w + α_c · δ · ∇w V(s)
#
#   ACTOR:   learns π(a|s) to select actions
#            Update:  θ ← θ + α_a · δ · ∇θ ln π(a|s)
#
#   TD ERROR (the "critic's verdict"):
#            δ = r + γ V(s') − V(s)
#
#   The TD error δ replaces the return Gₜ from REINFORCE.
#   This is what makes actor-critic LOWER VARIANCE than REINFORCE.
#
# ┌──────────┐     ┌──────────┐
# │  ACTOR   │     │  CRITIC  │
# │  π(a|s;θ)│     │  V(s; w) │
# └────┬─────┘     └────┬─────┘
#      │   action a      │
#      └────────┬────────┘
#               ▼
#         Environment
#         r, s' ──→ δ = r + γV(s') − V(s)
#                    │
#              ┌─────┴─────┐
#              ▼           ▼
#         Update θ    Update w

def sim1_one_step_actor_critic():
    print_header("SIM 1: One-Step Actor-Critic (Core Algorithm)")

    env = ChainWalk(n_states=7)
    N_STATES = env.n_states
    N_ACTIONS = 2
    GAMMA = 0.99
    ALPHA_ACTOR = 0.1     # learning rate for policy (actor)
    ALPHA_CRITIC = 0.2    # learning rate for value fn (critic)
    NUM_EPISODES = 300

    # ── CRITIC: weights for V(s) ──
    # V(s) = w ᵀ x(s)   where x(s) is one-hot
    # so w[s] = V(s)  (table lookup)
    w_critic = np.zeros(N_STATES)

    # ── ACTOR: weights for policy logits ──
    # logit(a|s) = θ[s, a]
    # π(a|s) = softmax(θ[s, :])
    theta_actor = np.zeros((N_STATES, N_ACTIONS))

    print(f"\n  States: {N_STATES}  |  Actions: {N_ACTIONS}")
    print(f"  α_actor={ALPHA_ACTOR}, α_critic={ALPHA_CRITIC}, γ={GAMMA}")
    print(f"\n  Architecture:")
    print(f"    Critic: V(s) = w[s]          (table lookup)")
    print(f"    Actor:  π(a|s) = softmax(θ[s,:])  (table lookup)")

    print(f"\n  --- First 3 episodes shown step-by-step ---\n")

    episode_rewards = []

    for ep in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        step = 0
        show = ep < 3  # show detail for first 3 episodes

        if show:
            print(f"  ┌─── Episode {ep+1} ───┐")

        while True:
            # ── ACTOR: select action from policy ──
            logits = theta_actor[state]          # θ[s, :]
            probs = softmax(logits)              # π(·|s)
            action = sample_action(probs)        # sample a ~ π(·|s)

            # ── Take action, observe reward and next state ──
            next_state, reward, done = env.step(action)
            total_reward += reward

            # ── CRITIC: compute TD error δ ──
            V_s = w_critic[state]                # V(s)
            V_next = 0.0 if done else w_critic[next_state]  # V(s')
            td_error = reward + GAMMA * V_next - V_s        # δ = r + γV(s') − V(s)

            # ── CRITIC UPDATE ──
            # w[s] ← w[s] + α_c · δ
            # (gradient of V(s) w.r.t. w[s] is 1 for table lookup)
            w_critic[state] += ALPHA_CRITIC * td_error

            # ── ACTOR UPDATE ──
            # ∇θ ln π(a|s) = x(s,a) − Σ_a' π(a'|s) x(s,a')
            # For softmax: ∇θ ln π(a|s) = e_a − π   (one-hot minus probs)
            #
            # θ ← θ + α_a · δ · ∇θ ln π(a|s)
            grad_log_pi = np.zeros(N_ACTIONS)
            grad_log_pi[action] = 1.0          # one-hot for chosen action
            grad_log_pi -= probs                # subtract probabilities

            # This is the KEY: δ scales the policy gradient!
            theta_actor[state] += ALPHA_ACTOR * td_error * grad_log_pi

            if show and step < 5:
                a_name = "Left " if action == 0 else "Right"
                print(f"  │ Step {step}: s={state} → a={a_name} → s'={next_state}, r={reward:+.1f}")
                print(f"  │   π(·|s={state}) = [{probs[0]:.3f}, {probs[1]:.3f}]")
                print(f"  │   V(s)={V_s:+.3f}, V(s')={V_next:+.3f}")
                print(f"  │   TD error δ = {reward:+.1f} + {GAMMA}×{V_next:+.3f} − {V_s:+.3f} = {td_error:+.4f}")
                print(f"  │   ∇ln π = {np.round(grad_log_pi, 3)}")
                print(f"  │   Critic update: w[{state}] += {ALPHA_CRITIC}×{td_error:+.4f} = {ALPHA_CRITIC*td_error:+.6f}")
                print(f"  │   Actor update:  θ[{state}] += {ALPHA_ACTOR}×{td_error:+.4f}×∇ln π")
                print(f"  │")

            step += 1
            state = next_state
            if done or step > 100:
                break

        episode_rewards.append(total_reward)
        if show:
            print(f"  └─── Total reward: {total_reward:+.1f} ({step} steps) ───┘\n")

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  Episode {ep+1:>4d}  |  Avg reward (last 50): {avg:+.3f}")

    # Print final learned values and policy
    print(f"\n  --- Learned Critic V(s) ---")
    for s in range(1, N_STATES - 1):
        bar = "█" * int(max(0, w_critic[s]) * 20)
        print(f"    V({s}) = {w_critic[s]:+.4f}  {bar}")

    print(f"\n  --- Learned Actor π(a|s) ---")
    for s in range(1, N_STATES - 1):
        probs = softmax(theta_actor[s])
        left_bar = "◀" * int(probs[0] * 20)
        right_bar = "▶" * int(probs[1] * 20)
        print(f"    s={s}: π(Left)={probs[0]:.3f} {left_bar:>20s} | {right_bar:<20s} π(Right)={probs[1]:.3f}")

    print(f"\n  Key insight: The CRITIC provides the TD error δ which tells the")
    print(f"  ACTOR whether the action was better or worse than expected.")
    print(f"  δ > 0 → action was better than V(s) predicted → increase its probability")
    print(f"  δ < 0 → action was worse → decrease its probability")


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION 2: Actor-Critic with Eligibility Traces
# ─────────────────────────────────────────────────────────────────────────────
#
# Problem with one-step AC: TD error only looks one step ahead.
# Solution: eligibility traces propagate credit further back.
#
#   Critic trace:  e_w ← γλ_w e_w + ∇w V(s)
#   Actor trace:   e_θ ← γλ_θ e_θ + ∇θ ln π(a|s)
#
#   Then:  w ← w + α_c · δ · e_w
#          θ ← θ + α_a · δ · e_θ
#
# The traces remember which states/actions were recently visited
# and assign them proportional credit when δ arrives.

def sim2_actor_critic_traces():
    print_header("SIM 2: Actor-Critic with Eligibility Traces")

    env = ChainWalk(n_states=9)
    N_STATES = env.n_states
    N_ACTIONS = 2
    GAMMA = 0.99
    LAMBDA_W = 0.8       # trace decay for critic
    LAMBDA_THETA = 0.8   # trace decay for actor
    ALPHA_ACTOR = 0.05
    ALPHA_CRITIC = 0.1
    NUM_EPISODES = 300

    w_critic = np.zeros(N_STATES)
    theta_actor = np.zeros((N_STATES, N_ACTIONS))

    print(f"\n  λ_critic={LAMBDA_W}, λ_actor={LAMBDA_THETA}")
    print(f"  Eligibility traces propagate TD error to past states")

    episode_rewards = []

    for ep in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0

        # ── Initialize eligibility traces to zero ──
        e_w = np.zeros(N_STATES)                    # critic trace
        e_theta = np.zeros((N_STATES, N_ACTIONS))   # actor trace

        show = ep == 0
        if show:
            print(f"\n  ┌─── Episode 1 (detailed trace inspection) ───┐")

        step = 0
        while True:
            logits = theta_actor[state]
            probs = softmax(logits)
            action = sample_action(probs)

            next_state, reward, done = env.step(action)
            total_reward += reward

            # TD error
            V_s = w_critic[state]
            V_next = 0.0 if done else w_critic[next_state]
            delta = reward + GAMMA * V_next - V_s

            # ── Update eligibility traces ──
            # Critic trace: accumulate gradient of V(s) for visited states
            e_w = GAMMA * LAMBDA_W * e_w          # decay old traces
            e_w[state] += 1.0                       # boost current state

            # Actor trace: accumulate ∇θ ln π(a|s) for visited (s,a)
            grad_log_pi = np.zeros(N_ACTIONS)
            grad_log_pi[action] = 1.0
            grad_log_pi -= probs
            e_theta = GAMMA * LAMBDA_THETA * e_theta   # decay
            e_theta[state] += grad_log_pi               # boost

            # ── Updates using traces ──
            w_critic += ALPHA_CRITIC * delta * e_w       # ALL traced states updated
            theta_actor += ALPHA_ACTOR * delta * e_theta # ALL traced (s,a) updated

            if show and step < 8:
                print(f"  │ Step {step}: s={state} a={'L' if action==0 else 'R'} → s'={next_state} r={reward:+.1f}")
                print(f"  │   δ = {delta:+.4f}")
                # Show which states have non-zero traces
                active_traces = [(i, e_w[i]) for i in range(N_STATES) if abs(e_w[i]) > 0.01]
                trace_str = ", ".join([f"e_w[{i}]={v:.3f}" for i, v in active_traces])
                print(f"  │   Active critic traces: {trace_str}")
                print(f"  │   → δ updates {len(active_traces)} states simultaneously!")
                print(f"  │")

            step += 1
            state = next_state
            if done or step > 100:
                break

        episode_rewards.append(total_reward)

        if show:
            print(f"  └─── Total reward: {total_reward:+.1f} ({step} steps) ───┘")

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"\n  Episode {ep+1:>4d}  |  Avg reward (last 50): {avg:+.3f}")

    print(f"\n  Key insight: Without traces (λ=0), only s is updated each step.")
    print(f"  With traces (λ={LAMBDA_W}), ALL recently visited states get credit.")
    print(f"  The trace decays by γλ each step → recent states get more credit.")
    print(f"  This bridges the gap between one-step TD and full Monte-Carlo.")


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION 3: Advantage Actor-Critic (A2C) on CartPole
# ─────────────────────────────────────────────────────────────────────────────
#
# Now with function approximation (not tables):
#
#   Critic:  V(s) = w ᵀ φ(s)       — linear in features
#   Actor:   π(a|s) = softmax(θ ᵀ φ(s))  — linear logits, softmax output
#
#   Advantage:  A(s,a) = δ = r + γV(s') − V(s)
#
# "Advantage" tells us: was this action better (+) or worse (−)
# than what the critic expected on average?

def sim3_a2c_cartpole():
    print_header("SIM 3: Advantage Actor-Critic (A2C) — CartPole")

    env = SimpleCartPole()
    N_FEATURES = 5     # [pos, vel, angle, ang_vel, bias]
    N_ACTIONS = 2      # push left / push right
    GAMMA = 0.99
    ALPHA_ACTOR = 0.005
    ALPHA_CRITIC = 0.01
    NUM_EPISODES = 500

    # Critic weights: V(s) = w_critic ᵀ φ(s)
    w_critic = np.zeros(N_FEATURES)

    # Actor weights: logit(a|s) = θ_actor[a] ᵀ φ(s)
    # One weight vector per action
    theta_actor = np.zeros((N_ACTIONS, N_FEATURES))

    print(f"\n  Features: {N_FEATURES}  |  Actions: {N_ACTIONS}")
    print(f"  Critic: V(s) = w ᵀ φ(s)           — linear value function")
    print(f"  Actor:  π(a|s) = softmax(θ ᵀ φ(s)) — linear policy\n")

    episode_lengths = []

    for ep in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        show = ep == 0

        if show:
            print(f"  ┌─── Episode 1 (first 5 steps) ───┐")

        step = 0
        while True:
            phi = env.get_features(state)    # feature vector φ(s)

            # ── ACTOR: compute policy ──
            logits = theta_actor @ phi        # [θ₀ᵀφ, θ₁ᵀφ]
            probs = softmax(logits)           # π(·|s)
            action = sample_action(probs)

            # ── Step environment ──
            next_state, reward, done = env.step(action)
            total_reward += reward

            phi_next = env.get_features(next_state)

            # ── CRITIC: TD error (= advantage estimate) ──
            V_s = w_critic @ phi                           # V(s) = wᵀφ(s)
            V_next = 0.0 if done else w_critic @ phi_next  # V(s')
            advantage = reward + GAMMA * V_next - V_s      # A(s,a) ≈ δ

            # ── CRITIC UPDATE ──
            # ∇w V(s) = φ(s)   for linear critic
            # w ← w + α_c · δ · φ(s)
            w_critic += ALPHA_CRITIC * advantage * phi

            # ── ACTOR UPDATE ──
            # Score function: ∇θ ln π(a|s)
            # For softmax linear policy:
            #   ∇θₐ ln π(a|s) = φ(s)  (for chosen action a)
            #   ∇θⱼ ln π(a|s) = −π(j|s) φ(s)  (for other actions j≠a)
            #
            # θ ← θ + α_a · advantage · ∇θ ln π(a|s)
            for a in range(N_ACTIONS):
                if a == action:
                    theta_actor[a] += ALPHA_ACTOR * advantage * (1 - probs[a]) * phi # ( ϕ(s,a) -prob[a]) ϕ(s,a) is 1 because it is the actual outcome?  
                else:
                    theta_actor[a] += ALPHA_ACTOR * advantage * (-probs[a]) * phi

            if show and step < 5:
                print(f"  │ Step {step}:")
                print(f"  │   state  = [{state[0]:+.3f}, {state[1]:+.3f}, {state[2]:+.3f}, {state[3]:+.3f}]")
                print(f"  │   φ(s)   = {np.round(phi, 3)}")
                print(f"  │   logits = {np.round(logits, 4)}")
                print(f"  │   π(·|s) = [{probs[0]:.4f}, {probs[1]:.4f}]")
                print(f"  │   action = {'Push-Left' if action==0 else 'Push-Right'}")
                print(f"  │   V(s)={V_s:+.4f}, V(s')={V_next:+.4f}")
                print(f"  │   Advantage δ = {reward} + {GAMMA}×{V_next:+.4f} − {V_s:+.4f} = {advantage:+.4f}")
                if advantage > 0:
                    print(f"  │   → Action was BETTER than expected → increase π(a|s)")
                else:
                    print(f"  │   → Action was WORSE than expected → decrease π(a|s)")
                print(f"  │")

            step += 1
            state = next_state
            if done:
                break

        episode_lengths.append(step)

        if show:
            print(f"  └─── Survived {step} steps ───┘\n")

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_lengths[-50:])
            bar = "█" * int(avg / 5)
            print(f"  Episode {ep+1:>4d}  |  Avg length (last 50): {avg:>6.1f}  {bar}")

    print(f"\n  Final critic weights w: {np.round(w_critic, 4)}")
    print(f"  (Large |w[2]| for angle means critic learned angle is important)")


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION 4: REINFORCE vs Actor-Critic Variance Comparison
# ─────────────────────────────────────────────────────────────────────────────
#
# REINFORCE:       Δθ = α · Gₜ · ∇θ ln π(a|s)
#   - Gₜ is the full return — UNBIASED but HIGH VARIANCE
#
# REINFORCE+baseline: Δθ = α · (Gₜ − b(s)) · ∇θ ln π(a|s)
#   - b(s) is a baseline that doesn't change the expected gradient
#   - but REDUCES VARIANCE
#
# Actor-Critic:    Δθ = α · δ · ∇θ ln π(a|s)
#   - δ = r + γV(s') − V(s) — BIASED but MUCH LOWER VARIANCE
#   - V(s) acts as both critic AND baseline
#
# This simulation runs both and compares gradient variance.

def sim4_reinforce_vs_ac():
    print_header("SIM 4: REINFORCE vs Actor-Critic — Variance Comparison")

    env = ChainWalk(n_states=7)
    N_STATES = env.n_states
    N_ACTIONS = 2
    GAMMA = 0.99
    ALPHA = 0.05
    NUM_EPISODES = 200

    # Run REINFORCE
    theta_rf = np.zeros((N_STATES, N_ACTIONS))
    reinforce_grad_norms = []
    reinforce_rewards = []

    # Run Actor-Critic
    theta_ac = np.zeros((N_STATES, N_ACTIONS))
    w_ac = np.zeros(N_STATES)
    ac_grad_norms = []
    ac_rewards = []

    print(f"\n  Comparing update signals:")
    print(f"    REINFORCE:     scale = Gₜ           (full return)")
    print(f"    Actor-Critic:  scale = δ = r + γV(s') − V(s)")
    print(f"\n  We measure gradient norm ‖Δθ‖ to compare variance.\n")

    for ep in range(NUM_EPISODES):
        # ── Generate episode ──
        state = env.reset()
        trajectory = []
        while True:
            probs = softmax(theta_rf[state])
            action = sample_action(probs)
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward, next_state, done))
            state = next_state
            if done or len(trajectory) > 100:
                break

        ep_reward = sum(r for _, _, r, _, _ in trajectory)
        reinforce_rewards.append(ep_reward)
        ac_rewards.append(ep_reward)  # same trajectory

        # ── REINFORCE update ──
        # Compute returns Gₜ for each timestep
        G = 0.0
        returns = []
        for t in reversed(range(len(trajectory))):
            G = trajectory[t][2] + GAMMA * G
            returns.insert(0, G)

        rf_total_grad = np.zeros_like(theta_rf)
        for t, (s, a, r, ns, d) in enumerate(trajectory):
            probs = softmax(theta_rf[s])
            grad_log_pi = np.zeros(N_ACTIONS)
            grad_log_pi[a] = 1.0
            grad_log_pi -= probs

            # REINFORCE: scale by FULL RETURN Gₜ
            rf_total_grad[s] += ALPHA * returns[t] * grad_log_pi

        theta_rf += rf_total_grad
        reinforce_grad_norms.append(np.linalg.norm(rf_total_grad))

        # ── Actor-Critic update ──
        ac_total_grad = np.zeros_like(theta_ac)
        for t, (s, a, r, ns, d) in enumerate(trajectory):
            probs = softmax(theta_ac[s])
            grad_log_pi = np.zeros(N_ACTIONS)
            grad_log_pi[a] = 1.0
            grad_log_pi -= probs

            V_s = w_ac[s]
            V_next = 0.0 if d else w_ac[ns]
            delta = r + GAMMA * V_next - V_s   # TD error

            # Actor-Critic: scale by TD ERROR δ (much smaller magnitude!)
            ac_total_grad[s] += ALPHA * delta * grad_log_pi

            # Update critic
            w_ac[s] += 0.1 * delta

        theta_ac += ac_total_grad
        ac_grad_norms.append(np.linalg.norm(ac_total_grad))

        if (ep + 1) % 50 == 0 or ep == 0:
            avg_rf = np.mean(reinforce_grad_norms[-20:])
            avg_ac = np.mean(ac_grad_norms[-20:])
            ratio = avg_rf / max(avg_ac, 1e-8)
            print(f"  Episode {ep+1:>4d}  |  "
                  f"REINFORCE ‖∇‖: {avg_rf:.4f}  |  "
                  f"Actor-Critic ‖∇‖: {avg_ac:.4f}  |  "
                  f"Ratio: {ratio:.1f}x")

    print(f"\n  Overall gradient norm statistics:")
    print(f"    REINFORCE     mean={np.mean(reinforce_grad_norms):.4f}  "
          f"std={np.std(reinforce_grad_norms):.4f}")
    print(f"    Actor-Critic  mean={np.mean(ac_grad_norms):.4f}  "
          f"std={np.std(ac_grad_norms):.4f}")
    print(f"\n  Key insight: Actor-Critic gradients have LOWER VARIANCE because")
    print(f"  the TD error δ = r + γV(s')−V(s) has much smaller magnitude than Gₜ.")
    print(f"  The critic V(s) acts as a baseline that centers the signal around zero.")


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION 5: Continuous Action Space Actor-Critic
# ─────────────────────────────────────────────────────────────────────────────
#
# For continuous actions, the actor outputs a GAUSSIAN DISTRIBUTION:
#   μ(s) = θ_μ ᵀ φ(s)        — mean depends on state
#   σ(s) = exp(θ_σ ᵀ φ(s))   — std also depends on state (or fixed)
#   a ~ N(μ(s), σ(s)²)
#
#   Score: ∇θ ln π(a|s) = ∇θ ln N(a; μ, σ²)
#     ∂/∂θ_μ = (a − μ) / σ² · φ(s)
#     ∂/∂θ_σ = ((a − μ)² / σ² − 1) · φ(s)
#
# Environment: reach target position on a 1D line

def sim5_continuous_action():
    print_header("SIM 5: Continuous Action Space Actor-Critic")

    # Simple 1D reaching task: state = position, action = velocity
    # Goal: reach target at position 5.0 starting from 0.0
    TARGET = 5.0
    GAMMA = 0.99
    ALPHA_ACTOR = 0.01
    ALPHA_CRITIC = 0.05
    SIGMA = 1.0          # fixed standard deviation for simplicity
    NUM_EPISODES = 300

    def get_features(pos):
        """Simple features: [position, distance_to_target, bias]"""
        return np.array([pos / 10.0, (TARGET - pos) / 10.0, 1.0])

    # Critic: V(s) = w ᵀ φ(s)
    w_critic = np.zeros(3)

    # Actor (mean only, fixed σ): μ(s) = θ_mu ᵀ φ(s)
    theta_mu = np.zeros(3)

    print(f"\n  Task: move from position 0 to target {TARGET}")
    print(f"  Action = continuous velocity, sampled from N(μ(s), σ²)")
    print(f"  σ = {SIGMA} (fixed)\n")

    for ep in range(NUM_EPISODES):
        pos = 0.0
        total_reward = 0
        show = ep < 2

        if show:
            print(f"  ┌─── Episode {ep+1} ───┐")

        for step in range(50):
            phi = get_features(pos)

            # ── ACTOR: sample continuous action from Gaussian ──
            mu = theta_mu @ phi                   # μ(s) = θᵀφ(s)
            action = np.random.normal(mu, SIGMA)  # a ~ N(μ, σ²)

            # ── Environment step ──
            new_pos = pos + action * 0.3  # scale down for stability
            distance = abs(new_pos - TARGET)
            reward = -distance * 0.1      # closer = higher reward
            done = distance < 0.3

            if done:
                reward += 5.0

            # ── CRITIC: TD error ──
            phi_next = get_features(new_pos)
            V_s = w_critic @ phi
            V_next = 0.0 if done else w_critic @ phi_next
            delta = reward + GAMMA * V_next - V_s

            # ── CRITIC UPDATE ──
            w_critic += ALPHA_CRITIC * delta * phi

            # ── ACTOR UPDATE (Gaussian policy gradient) ──
            # ∇θ_μ ln N(a; μ, σ²) = (a − μ) / σ² · φ(s)
            #
            # Intuition: if δ > 0 and a > μ, then push μ higher
            #            (the action was good and it was above average)
            score = (action - mu) / (SIGMA ** 2) * phi  # ∇θ ln π
            theta_mu += ALPHA_ACTOR * delta * score

            if show and step < 4:
                print(f"  │ Step {step}: pos={pos:.2f}")
                print(f"  │   μ(s) = {mu:.3f}, action = {action:.3f} (sampled from N({mu:.3f}, {SIGMA}²))")
                print(f"  │   → new_pos = {new_pos:.2f}, reward = {reward:.3f}")
                print(f"  │   δ = {delta:+.4f}")
                print(f"  │   Score ∇ln π = (a−μ)/σ² · φ = ({action:.3f}−{mu:.3f})/{SIGMA}² · φ")
                print(f"  │   θ_μ update = α · δ · score = {ALPHA_ACTOR}×{delta:+.3f}×score")
                print(f"  │")

            total_reward += reward
            pos = new_pos
            if done:
                break

        if show:
            print(f"  └─── Reward: {total_reward:+.1f}, final pos: {pos:.2f} ───┘\n")

        if (ep + 1) % 100 == 0:
            # Test greedy policy
            test_pos = 0.0
            for _ in range(50):
                phi = get_features(test_pos)
                mu = theta_mu @ phi
                test_pos += mu * 0.3
                if abs(test_pos - TARGET) < 0.3:
                    break
            print(f"  Episode {ep+1:>4d}  |  Greedy test: final pos = {test_pos:.2f} "
                  f"(target={TARGET}), θ_μ = {np.round(theta_mu, 3)}")

    print(f"\n  Key insight: For continuous actions, the actor outputs")
    print(f"  parameters of a distribution (here μ of a Gaussian).")
    print(f"  The score function (a−μ)/σ² tells us: if the sampled action")
    print(f"  was above/below the mean, push the mean in that direction")
    print(f"  proportionally to how good it was (δ).")


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION 6: Entropy-Regularised Actor-Critic
# ─────────────────────────────────────────────────────────────────────────────
#
# Problem: policy can collapse to always choosing one action (no exploration).
# Solution: add entropy bonus to encourage exploration.
#
#   H(π(·|s)) = − Σ_a π(a|s) ln π(a|s)
#
#   Modified objective: maximise  E[Σ γᵗ (rₜ + β H(π(·|sₜ)))]
#
#   Actor update becomes:
#     θ ← θ + α · [δ · ∇θ ln π + β · ∇θ H(π)]
#
#   Higher entropy = more uniform policy = more exploration

def sim6_entropy_regularised():
    print_header("SIM 6: Entropy-Regularised Actor-Critic")

    env = ChainWalk(n_states=7)
    N_STATES = env.n_states
    N_ACTIONS = 2
    GAMMA = 0.99
    ALPHA_ACTOR = 0.1
    ALPHA_CRITIC = 0.2
    NUM_EPISODES = 300

    def compute_entropy(probs):
        """H(π) = −Σ π(a) ln π(a)"""
        return -sum(p * math.log(p + 1e-10) for p in probs)

    def entropy_gradient(probs):
        """∇θ H(π) for softmax policy.
        = −Σ_a ∇θ[π(a) ln π(a)]
        = −Σ_a [∇θ π(a) · (1 + ln π(a))]
        For softmax: ∇θ_i π(a) = π(a)(δ_{ia} − π(i))
        """
        grad = np.zeros(N_ACTIONS)
        for a in range(N_ACTIONS):
            for i in range(N_ACTIONS):
                indicator = 1.0 if i == a else 0.0
                grad[i] -= probs[a] * (indicator - probs[i]) * (1 + math.log(probs[a] + 1e-10))
        return grad

    # Run with and without entropy bonus
    for beta, label in [(0.0, "No entropy bonus (β=0)"), (0.1, "With entropy bonus (β=0.1)")]:
        print(f"\n  ── {label} ──")

        w = np.zeros(N_STATES)
        theta = np.zeros((N_STATES, N_ACTIONS))
        episode_rewards = []
        entropies = []

        for ep in range(NUM_EPISODES):
            state = env.reset()
            total_reward = 0
            ep_entropy = []

            while True:
                probs = softmax(theta[state])
                action = sample_action(probs)
                next_state, reward, done = env.step(action)
                total_reward += reward

                V_s = w[state]
                V_next = 0.0 if done else w[next_state]
                delta = reward + GAMMA * V_next - V_s

                # Critic update (same as before)
                w[state] += ALPHA_CRITIC * delta

                # Actor update WITH entropy bonus
                grad_log_pi = np.zeros(N_ACTIONS)
                grad_log_pi[action] = 1.0
                grad_log_pi -= probs

                # Standard policy gradient + entropy gradient
                grad_H = entropy_gradient(probs)
                theta[state] += ALPHA_ACTOR * (delta * grad_log_pi + beta * grad_H)

                H = compute_entropy(probs)
                ep_entropy.append(H)

                state = next_state
                if done or len(ep_entropy) > 100:
                    break

            episode_rewards.append(total_reward)
            entropies.append(np.mean(ep_entropy))

            if (ep + 1) % 100 == 0:
                avg_r = np.mean(episode_rewards[-50:])
                avg_h = np.mean(entropies[-50:])
                print(f"    Episode {ep+1:>4d}  |  Avg reward: {avg_r:+.3f}  |  "
                      f"Avg entropy: {avg_h:.4f} (max={math.log(N_ACTIONS):.4f})")

        # Show final policies
        print(f"    Final policy:")
        for s in range(1, N_STATES - 1):
            probs = softmax(theta[s])
            H = compute_entropy(probs)
            print(f"      s={s}: π(L)={probs[0]:.3f}, π(R)={probs[1]:.3f}  "
                  f"H={H:.3f}  {'← exploring' if H > 0.4 else '← exploiting'}")

    print(f"\n  Key insight: Entropy bonus β·H(π) prevents premature convergence.")
    print(f"  Without it, π quickly collapses to deterministic (H≈0).")
    print(f"  With β > 0, the agent maintains some randomness, exploring more,")
    print(f"  which often leads to better long-term performance.")
    print(f"  This is used in SAC, A3C, and many modern algorithms.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  Actor-Critic Methods — Step-by-Step Python Simulations         ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                 ║
    ║    ACTOR  (policy π(a|s; θ))      CRITIC  (value V(s; w))      ║
    ║    "What should I do?"            "How good is this state?"     ║
    ║                                                                 ║
    ║    Both learn simultaneously:                                   ║
    ║    - Critic evaluates → produces TD error δ                    ║
    ║    - Actor improves  → uses δ as learning signal               ║
    ║                                                                 ║
    ║    δ > 0 → action was better than expected → reinforce it      ║
    ║    δ < 0 → action was worse  than expected → discourage it     ║
    ║                                                                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    sim1_one_step_actor_critic()
    sim2_actor_critic_traces()
    sim3_a2c_cartpole()
    sim4_reinforce_vs_ac()
    sim5_continuous_action()
    sim6_entropy_regularised()

    print("\n" + "=" * 72)
    print("  ALL SIMULATIONS COMPLETE")
    print("=" * 72)
    print("""
  Summary of Actor-Critic concepts demonstrated:

  Sim 1 — One-Step AC:    The core loop. Critic gives δ, actor uses it.
           δ = r + γV(s') − V(s)
           Actor:  θ ← θ + α · δ · ∇θ ln π(a|s)
           Critic: w ← w + α · δ · ∇w V(s)

  Sim 2 — Eligibility Traces:  Propagate credit to past states.
           e ← γλe + ∇    (trace remembers recent visits)
           Updates use δ · e  (all traced states updated at once)

  Sim 3 — A2C on CartPole:  Function approximation (not tables).
           Critic: V(s) = wᵀφ(s),  Actor: π = softmax(θᵀφ(s))
           Advantage A(s,a) ≈ δ = r + γV(s') − V(s)

  Sim 4 — REINFORCE vs AC:  Why the critic helps.
           REINFORCE uses Gₜ (high variance), AC uses δ (low variance).
           Lower variance → more stable, faster learning.

  Sim 5 — Continuous Actions:  Gaussian policy π = N(μ(s), σ²).
           Score: ∇ln π = (a−μ)/σ² · φ(s)
           "If action above mean worked well, push mean up"

  Sim 6 — Entropy Regularisation:  H(π) = −Σ π ln π
           Bonus β·H prevents policy collapse, maintains exploration.
           Used in SAC, A3C, and most modern methods.
    """)
