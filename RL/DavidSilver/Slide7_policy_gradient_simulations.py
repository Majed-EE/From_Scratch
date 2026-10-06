"""
=============================================================================
  Policy Gradient Methods — David Silver Lecture 7
  Step-by-Step Python Simulations
=============================================================================

Run:  python policy_gradient_simulations.py

Simulations:
  1. Finite Difference Policy Gradient   (numerical, no calculus needed)
  2. Score Function & Softmax Policy      (analytical gradient, likelihood ratio)
  3. REINFORCE (Monte-Carlo Policy Grad)  (the classic algorithm)
  4. REINFORCE with Baseline              (variance reduction via V(s))
  5. Actor-Critic (TD Actor-Critic)       (critic replaces MC return)
  6. Advantage Actor-Critic (A2C)         (advantage = Q - V ≈ δ)
  7. Natural Policy Gradient              (Fisher information matrix)
  8. Summary: All variants head-to-head   (convergence + variance comparison)
=============================================================================
"""

import numpy as np
import math
import random

np.random.seed(42)
random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Helpers
# ─────────────────────────────────────────────────────────────────────────────

def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    logits -= logits.max() #why did we do this: for numerical stability to prevent large exponentials
    e = np.exp(logits)
    return e / e.sum()

def sample(probs):
    return np.random.choice(len(probs), p=probs)

def hdr(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


# ─────────────────────────────────────────────────────────────────────────────
# Environment: Short Corridor with Switched Actions
# ─────────────────────────────────────────────────────────────────────────────
# States: 0(start) — 1 — 2 — 3(goal)
# Actions: 0=left, 1=right
# Twist: in state 1, actions are REVERSED (left goes right, right goes left)
# Reward: -1 per step, 0 at goal
# This environment needs a STOCHASTIC policy to solve well
# (deterministic "always right" gets stuck oscillating at state 1)

class ShortCorridor:
    """From Sutton & Barto Example 13.1 — needs stochastic policy."""
    def __init__(self):
        self.n_states = 4
        self.goal = 3

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        s = self.state
        # In state 1, actions are reversed!
        if s == 1:
            actual = 1 - action
        else:
            actual = action

        if actual == 0:  # left
            self.state = max(0, s - 1)
        else:            # right
            self.state = min(3, s + 1)

        done = self.state == self.goal
        reward = 0.0 if done else -1.0
        return self.state, reward, done


# ─────────────────────────────────────────────────────────────────────────────
# Environment: Chain Walk (for multi-state demos)
# ─────────────────────────────────────────────────────────────────────────────

class ChainWalk:
    def __init__(self, n=7):
        self.n = n
    def reset(self):
        self.s = self.n // 2
        return self.s
    def step(self, a):
        self.s = max(0, min(self.n-1, self.s + (1 if a==1 else -1)))
        if self.s == self.n-1: return self.s, +1.0, True
        if self.s == 0:        return self.s, -1.0, True
        return self.s, 0.0, False


# ─────────────────────────────────────────────────────────────────────────────
# SIM 1: Finite Difference Policy Gradient
# ─────────────────────────────────────────────────────────────────────────────
#
# Slide: "Computing Gradients By Finite Differences"
#
#   ∂J(θ)/∂θ_k ≈ [J(θ + ε·u_k) − J(θ)] / ε
#
# - Perturb each parameter by ε, measure change in J
# - No calculus needed — works for ANY policy (even non-differentiable)
# - But: noisy, slow (n evaluations for n parameters)

def sim1_finite_difference():
    hdr("SIM 1: Finite Difference Policy Gradient")

    env = ShortCorridor()
    N_PARAMS = 1  # single param: logit for "go right"

    def policy_probs(theta):
        """π(right|s) = sigmoid(θ), π(left|s) = 1 - sigmoid(θ)"""
        p_right = 1.0 / (1.0 + np.exp(-theta[0]))
        return np.array([1 - p_right, p_right])

    def evaluate_J(theta, n_episodes=200):
        """Estimate J(θ) by averaging returns over many episodes."""
        total = 0.0
        for _ in range(n_episodes):
            s = env.reset()
            ret = 0.0
            for _ in range(50):
                probs = policy_probs(theta)
                a = sample(probs)
                s, r, done = env.step(a)
                ret += r
                if done: break
            total += ret
        return total / n_episodes

    theta = np.array([0.0])  # start: 50-50 policy
    EPSILON = 0.5
    ALPHA = 2.0
    N_ITERS = 30

    print(f"\n  Method: perturb θ by ε={EPSILON}, measure ΔJ, compute gradient")
    print(f"  ∂J/∂θ ≈ [J(θ+ε) − J(θ)] / ε")
    print(f"\n  {'Iter':>4s}  {'θ':>7s}  {'π(R)':>6s}  {'J(θ)':>8s}  {'∂J/∂θ':>8s}")
    print(f"  " + "─" * 42)

    for i in range(N_ITERS):
        J_current = evaluate_J(theta)
        probs = policy_probs(theta)

        # ── Finite difference gradient ──
        # Perturb θ in the +ε direction
        J_perturbed = evaluate_J(theta + np.array([EPSILON]))
        grad = (J_perturbed - J_current) / EPSILON

        if i < 10 or i % 5 == 0:
            print(f"  {i:>4d}  {theta[0]:>+7.3f}  {probs[1]:>6.3f}  "
                  f"{J_current:>+8.2f}  {grad:>+8.4f}")

        # ── Gradient ASCENT (maximise J) ──
        theta = theta + ALPHA * np.array([grad])

    print(f"\n  Final: θ={theta[0]:.3f}, π(right)={policy_probs(theta)[1]:.3f}")
    print(f"\n  Key insight: No derivatives needed! Just evaluate J at θ and θ+ε.")
    print(f"  Works for ANY policy. But noisy and needs n evaluations for n params.")


# ─────────────────────────────────────────────────────────────────────────────
# SIM 2: Score Function & Softmax Policy
# ─────────────────────────────────────────────────────────────────────────────
#
# Slide: "Score Function" + "Softmax Policy"
#
#   Likelihood ratio trick:
#     ∇θ π(a|s) = π(a|s) · ∇θ log π(a|s)
#
#   For softmax policy:  π(a|s) ∝ exp(φ(s,a)ᵀθ)
#     Score function: ∇θ log π(a|s) = φ(s,a) − E_π[φ(s,·)]
#
#   This is the ANALYTICAL gradient — no perturbation needed.

def sim2_score_function():
    hdr("SIM 2: Score Function & Softmax Policy")

    # Simple example: 2 actions, 3 features per (s,a)
    N_FEATURES = 3
    N_ACTIONS = 2

    theta = np.array([0.5, -0.2, 0.3])

    # Feature vectors for state s=0
    phi = np.array([
        [1.0, 0.0, 0.5],   # φ(s, a=0)  "go left"
        [0.0, 1.0, 0.5],   # φ(s, a=1)  "go right"
    ])

    print(f"\n  Softmax policy: π(a|s) ∝ exp(φ(s,a)ᵀθ)")
    print(f"\n  θ = {theta}")
    print(f"  φ(s, left)  = {phi[0]}")
    print(f"  φ(s, right) = {phi[1]}")

    # ── Compute policy probabilities ──
    logits = phi @ theta  # [φ(s,a=0)ᵀθ, φ(s,a=1)ᵀθ]
    probs = softmax(logits)

    print(f"\n  Logits: [{logits[0]:.3f}, {logits[1]:.3f}]")
    print(f"  π(left|s)  = {probs[0]:.4f}")
    print(f"  π(right|s) = {probs[1]:.4f}")

    # ── Score function for each action ──
    # ∇θ log π(a|s) = φ(s,a) − Σ_a' π(a'|s) φ(s,a')
    expected_phi = probs @ phi   # E_π[φ(s,·)]
    print(f"\n  E_π[φ(s,·)] = {np.round(expected_phi, 4)}")

    for a, aname in enumerate(["left", "right"]):
        score = phi[a] - expected_phi
        print(f"\n  Score for a={aname}:")
        print(f"    ∇θ log π({aname}|s) = φ(s,{aname}) − E_π[φ(s,·)]")
        print(f"                        = {phi[a]} − {np.round(expected_phi, 4)}")
        print(f"                        = {np.round(score, 4)}")

    # ── Verify: E_π[score] = 0 (scores are centered) ──
    expected_score = sum(probs[a] * (phi[a] - expected_phi) for a in range(N_ACTIONS))
    print(f"\n  Verification: E_π[∇θ log π] = {np.round(expected_score, 8)}")
    print(f"  (Should be ≈ 0 — scores are automatically centered!)")

    # ── Policy gradient for a one-step problem ──
    rewards = np.array([1.0, 3.0])  # R(s, left)=1, R(s, right)=3
    print(f"\n  If rewards are: R(left)={rewards[0]}, R(right)={rewards[1]}")

    policy_gradient = np.zeros(N_FEATURES)
    for a in range(N_ACTIONS):
        score = phi[a] - expected_phi
        policy_gradient += probs[a] * score * rewards[a]

    print(f"  ∇θ J(θ) = Σ_a π(a|s) · ∇θ log π(a|s) · R(s,a)")
    print(f"          = {np.round(policy_gradient, 4)}")
    print(f"\n  This gradient points toward parameters that increase π(right)")
    print(f"  because right has higher reward (3 > 1).")


# ─────────────────────────────────────────────────────────────────────────────
# SIM 3: REINFORCE (Monte-Carlo Policy Gradient)
# ─────────────────────────────────────────────────────────────────────────────
#
# Slide: "Monte-Carlo Policy Gradient (REINFORCE)"
#
#   Δθ = α · ∇θ log π(sₜ, aₜ) · vₜ
#
#   where vₜ = Gₜ = Σ γᵏ rₜ₊ₖ  (full MC return)
#
# - Uses complete episode returns (unbiased but high variance)
# - Update AFTER the episode ends (need full trajectory)

def sim3_reinforce():
    hdr("SIM 3: REINFORCE (Monte-Carlo Policy Gradient)")

    env = ChainWalk(n=7)
    N_STATES = 7
    N_ACTIONS = 2
    GAMMA = 0.99
    ALPHA = 0.05
    NUM_EPISODES = 500

    # θ[s, a] = logit for action a in state s
    theta = np.zeros((N_STATES, N_ACTIONS))

    print(f"\n  REINFORCE algorithm (from the slides):")
    print(f"    1. Generate episode using π_θ")
    print(f"    2. For each step t:")
    print(f"       Compute return Gₜ = Σ γᵏ rₜ₊ₖ")
    print(f"       θ ← θ + α · ∇θ log π(sₜ,aₜ) · Gₜ")
    print(f"\n  Key: Gₜ is an UNBIASED sample of Q^π(s,a)")
    print(f"       But it has HIGH VARIANCE (whole episode of randomness)\n")

    rewards_history = []
    grad_norms = []

    for ep in range(NUM_EPISODES):
        # ── 1. Generate episode ──
        s = env.reset()
        trajectory = []
        while True:
            probs = softmax(theta[s])
            a = sample(probs)
            s_next, r, done = env.step(a)
            trajectory.append((s, a, r))
            s = s_next
            if done or len(trajectory) > 100: break

        # ── 2. Compute returns Gₜ for each step ──
        T = len(trajectory)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = trajectory[t][2] + GAMMA * G
            returns[t] = G

        # ── 3. Update θ using policy gradient ──
        total_grad_norm = 0.0
        for t in range(T):
            s_t, a_t, _ = trajectory[t]
            G_t = returns[t]

            # Score function for softmax
            probs = softmax(theta[s_t])
            score = np.zeros(N_ACTIONS)
            score[a_t] = 1.0
            score -= probs   # ∇θ log π(a|s) = e_a − π

            # REINFORCE update: θ ← θ + α · score · Gₜ
            theta[s_t] += ALPHA * score * G_t
            total_grad_norm += np.linalg.norm(score * G_t)

        ep_reward = sum(r for _, _, r in trajectory)
        rewards_history.append(ep_reward)
        grad_norms.append(total_grad_norm / max(T, 1))

        if ep < 3:
            print(f"  Episode {ep+1}: {T} steps, return={ep_reward:+.1f}")
            if ep == 0:
                print(f"    Step 0: s={trajectory[0][0]}, a={'R' if trajectory[0][1] else 'L'}, "
                      f"G₀={returns[0]:+.3f}")
                print(f"    Step 1: s={trajectory[1][0]}, a={'R' if trajectory[1][1] else 'L'}, "
                      f"G₁={returns[1]:+.3f}")

        if (ep+1) % 100 == 0:
            avg_r = np.mean(rewards_history[-50:])
            avg_gn = np.mean(grad_norms[-50:])
            print(f"  Episode {ep+1:>4d}  |  Avg reward: {avg_r:+.3f}  |  "
                  f"Avg |∇|: {avg_gn:.4f}")

    print(f"\n  Final policy (π(right|s)):")
    for s in range(1, N_STATES-1):
        p = softmax(theta[s])
        bar = "▶" * int(p[1] * 20)
        print(f"    s={s}: π(R)={p[1]:.3f}  {bar}")

    return rewards_history, grad_norms


# ─────────────────────────────────────────────────────────────────────────────
# SIM 4: REINFORCE with Baseline
# ─────────────────────────────────────────────────────────────────────────────
#
# Slide: "Reducing Variance Using a Baseline"
#
#   Δθ = α · ∇θ log π(sₜ, aₜ) · (Gₜ − b(sₜ))
#
#   b(s) = V(s) is a good baseline
#   A(s,a) = Q(s,a) − V(s) is the advantage function
#
# - Subtracting baseline does NOT change expected gradient
# - But REDUCES VARIANCE (centers the signal around zero)

def sim4_reinforce_baseline():
    hdr("SIM 4: REINFORCE with Baseline (Variance Reduction)")

    env = ChainWalk(n=7)
    N_STATES = 7
    N_ACTIONS = 2
    GAMMA = 0.99
    ALPHA_ACTOR = 0.05
    ALPHA_BASELINE = 0.1
    NUM_EPISODES = 500

    theta = np.zeros((N_STATES, N_ACTIONS))
    V = np.zeros(N_STATES)  # baseline b(s) = V(s)

    print(f"\n  REINFORCE + Baseline:")
    print(f"    Δθ = α · ∇θ log π(s,a) · (Gₜ − V(s))  ← advantage estimate")
    print(f"    V(s) ← V(s) + α_b · (Gₜ − V(s))       ← learn baseline")
    print(f"\n  WHY baseline helps:")
    print(f"    Without baseline: Gₜ might be +0.8 for all actions")
    print(f"      → all actions get reinforced, gradient is noisy")
    print(f"    With baseline:    Gₜ − V(s) is +0.3 for good, -0.2 for bad")
    print(f"      → clear signal about which action is BETTER than average\n")

    rewards_history = []
    grad_norms = []

    for ep in range(NUM_EPISODES):
        s = env.reset()
        trajectory = []
        while True:
            probs = softmax(theta[s])
            a = sample(probs)
            s_next, r, done = env.step(a)
            trajectory.append((s, a, r))
            s = s_next
            if done or len(trajectory) > 100: break

        T = len(trajectory)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = trajectory[t][2] + GAMMA * G
            returns[t] = G

        total_gn = 0.0
        for t in range(T):
            s_t, a_t, _ = trajectory[t]
            G_t = returns[t]

            # ── Advantage estimate: Gₜ − V(sₜ) ──
            advantage = G_t - V[s_t]

            # ── Update baseline ──
            V[s_t] += ALPHA_BASELINE * (G_t - V[s_t])

            # ── Actor update with baseline ──
            probs = softmax(theta[s_t])
            score = np.zeros(N_ACTIONS)
            score[a_t] = 1.0
            score -= probs

            theta[s_t] += ALPHA_ACTOR * score * advantage
            total_gn += np.linalg.norm(score * advantage)

        ep_reward = sum(r for _, _, r in trajectory)
        rewards_history.append(ep_reward)
        grad_norms.append(total_gn / max(T, 1))

        if (ep+1) % 100 == 0:
            avg_r = np.mean(rewards_history[-50:])
            avg_gn = np.mean(grad_norms[-50:])
            print(f"  Episode {ep+1:>4d}  |  Avg reward: {avg_r:+.3f}  |  "
                  f"Avg |∇|: {avg_gn:.4f}")

    print(f"\n  Learned baseline V(s):")
    for s in range(1, N_STATES-1):
        print(f"    V({s}) = {V[s]:+.3f}")

    return rewards_history, grad_norms


# ─────────────────────────────────────────────────────────────────────────────
# SIM 5: Actor-Critic (TD Actor-Critic)
# ─────────────────────────────────────────────────────────────────────────────
#
# Slide: "Action-Value Actor-Critic" + "Estimating the Advantage Function (2)"
#
#   δ = r + γ V(s') − V(s)          ← TD error
#   w ← w + β · δ · ∇w V(s)        ← critic update
#   θ ← θ + α · ∇θ log π(s,a) · δ  ← actor update
#
# - δ is a BIASED but LOW VARIANCE estimate of advantage
# - Updates every step (online), not at end of episode
# - The TD error δ replaces the MC return Gₜ

def sim5_actor_critic(): # wher is w and theta 
    hdr("SIM 5: Actor-Critic (TD Actor-Critic)")

    env = ChainWalk(n=7)
    N = 7
    GAMMA = 0.99
    ALPHA_ACTOR = 0.05
    ALPHA_CRITIC = 0.1
    NUM_EPISODES = 500

    theta = np.zeros((N, 2))
    V = np.zeros(N)  # critic

    print(f"\n  Actor-Critic algorithm (from the slides):")
    print(f"    CRITIC: δ = r + γV(s') − V(s)  then  V(s) += β·δ")
    print(f"    ACTOR:  θ += α · ∇θ log π(s,a) · δ")
    print(f"\n  Key difference from REINFORCE:")
    print(f"    REINFORCE uses Gₜ (full return, unbiased, high variance)")
    print(f"    Actor-Critic uses δ (TD error, biased, LOW variance)")
    print(f"    → Updates every STEP, not at end of episode\n")

    rewards_history = []
    grad_norms = []

    for ep in range(NUM_EPISODES):
        s = env.reset()
        ep_reward = 0
        ep_gn = 0
        steps = 0
        show = ep == 0

        if show:
            print(f"  ┌─── Episode 1 (step-by-step) ───┐")

        while True:

            # ϕ(s,a) is θ theta[s]  
            # ϕ(s,a) is softmax(theta[s])
            probs = softmax(theta[s]) # π θ (a,s) is this 
            # convert logits to probabilities using softmax for action selection
            a = sample(probs) 
            s_next, r, done = env.step(a)
            ep_reward += r
            steps += 1   #  

            # ── TD error (critic's verdict) ──
            V_next = 0.0 if done else V[s_next]
            delta = r + GAMMA * V_next - V[s]

            # ── Critic update ──
            V[s] += ALPHA_CRITIC * delta #  w ← w + β · δ · ∇w V(s) | ∇w V(s) = 1
            # where is w

            # ── Actor update ──
            score = np.zeros(2) # ∇θ log π(s,a)-> score |∇θ log π(s,a) = π(s,a) -E[ϕ(*,a)] | θ ← θ + α · ∇θ log π(s,a) · δ  
            score[a] = 1.0
            score -= probs # ∇θ log π(s,a) = π(s,a) -E[ϕ(*,a)] 
            theta[s] += ALPHA_ACTOR * score * delta # where is r equivalent
            ep_gn += abs(delta)

            if show and steps <= 5:
                aname = "R" if a else "L"
                print(f"  │ s={s} a={aname} → s'={s_next} r={r:+.1f}")
                print(f"  │   δ = {r:+.1f} + {GAMMA}·V({s_next})−V({s}) = {delta:+.4f}")
                print(f"  │   V({s}) += {ALPHA_CRITIC}·{delta:+.4f}")
                print(f"  │   θ[{s}] += {ALPHA_ACTOR}·score·{delta:+.4f}")
                if delta > 0:
                    print(f"  │   → δ>0: action was BETTER than expected")
                else:
                    print(f"  │   → δ<0: action was WORSE than expected")
                print(f"  │")

            s = s_next
            if done or steps > 100: break

        if show:
            print(f"  └─── reward={ep_reward:+.1f} ({steps} steps) ───┘\n")

        rewards_history.append(ep_reward)
        grad_norms.append(ep_gn / max(steps, 1))

        if (ep+1) % 100 == 0:
            avg_r = np.mean(rewards_history[-50:])
            print(f"  Episode {ep+1:>4d}  |  Avg reward: {avg_r:+.3f}")

    return rewards_history, grad_norms


# ─────────────────────────────────────────────────────────────────────────────
# SIM 6: Advantage Actor-Critic (A2C) with Eligibility Traces
# ─────────────────────────────────────────────────────────────────────────────
#
# Slide: "Policy Gradient with Eligibility Traces"
#
#   δ = rₜ₊₁ + γV(sₜ₊₁) − V(sₜ)
#   eₜ = λeₜ₋₁ + ∇θ log π(sₜ, aₜ)    ← actor trace
#   Δθ = α · δ · eₜ
#
# The trace remembers past (s,a) visits and assigns them credit.

def sim6_a2c_traces():
    hdr("SIM 6: Advantage Actor-Critic with Eligibility Traces")

    env = ChainWalk(n=7)
    N = 7
    GAMMA = 0.99
    LAMBDA = 0.8
    ALPHA_A = 0.03
    ALPHA_C = 0.1
    NUM_EPISODES = 500

    theta = np.zeros((N, 2))
    V = np.zeros(N)

    print(f"\n  TD(λ) Actor-Critic:")
    print(f"    δ = r + γV(s') − V(s)")
    print(f"    e_actor  = γλ·e_actor + ∇θ log π(s,a)   ← trace")
    print(f"    e_critic = γλ·e_critic + 1               ← trace")
    print(f"    θ += α_a · δ · e_actor")
    print(f"    V += α_c · δ · e_critic")
    print(f"\n  λ={LAMBDA}: traces propagate δ to past states\n")

    rewards_history = []

    for ep in range(NUM_EPISODES):
        s = env.reset()
        # ── Eligibility traces ──
        e_theta = np.zeros((N, 2))  # actor trace
        e_v = np.zeros(N)           # critic trace
        ep_reward = 0
        steps = 0

        while True:
            probs = softmax(theta[s])
            a = sample(probs)
            s_next, r, done = env.step(a)
            ep_reward += r
            steps += 1

            V_next = 0.0 if done else V[s_next]
            delta = r + GAMMA * V_next - V[s]

            # ── Update traces ──
            score = np.zeros(2)
            score[a] = 1.0
            score -= probs

            e_theta = GAMMA * LAMBDA * e_theta
            e_theta[s] += score       # boost current (s,a)

            e_v = GAMMA * LAMBDA * e_v
            e_v[s] += 1.0             # boost current state

            # ── Updates using traces (ALL traced states updated) ──
            theta += ALPHA_A * delta * e_theta
            V += ALPHA_C * delta * e_v

            s = s_next
            if done or steps > 100: break

        rewards_history.append(ep_reward)

        if (ep+1) % 100 == 0:
            avg_r = np.mean(rewards_history[-50:])
            active = np.sum(np.abs(e_v) > 0.01)
            print(f"  Episode {ep+1:>4d}  |  Avg reward: {avg_r:+.3f}  |  "
                  f"Active traces at end: {active}")

    return rewards_history


# ─────────────────────────────────────────────────────────────────────────────
# SIM 7: Natural Policy Gradient
# ─────────────────────────────────────────────────────────────────────────────
#
# Slide: "Natural Policy Gradient"
#
#   ∇_nat θ = G_θ⁻¹ · ∇θ J(θ)
#
#   G_θ = E[∇θ log π · (∇θ log π)ᵀ]   ← Fisher information matrix
#
# The natural gradient is parametrisation-independent.
# It measures "how much does the POLICY change" not "how much do params change".

def sim7_natural_gradient():
    hdr("SIM 7: Natural Policy Gradient")

    env = ChainWalk(n=7)
    N = 7
    GAMMA = 0.99
    ALPHA_VANILLA = 0.05
    ALPHA_NATURAL = 0.01
    NUM_EPISODES = 400

    print(f"\n  Natural gradient: ∇_nat = F⁻¹ · ∇J(θ)")
    print(f"  where F = E[∇log π · (∇log π)ᵀ] is the Fisher information matrix")
    print(f"\n  Vanilla gradient can be distorted by parametrisation.")
    print(f"  Natural gradient steps in the 'true' steepest direction")
    print(f"  in policy space (measured by KL divergence).\n")
    print(f"  Running vanilla vs natural gradient...\n")

    results = {}
    for method in ["Vanilla", "Natural"]:
        theta = np.zeros((N, 2))
        V = np.zeros(N)
        rewards = []

        for ep in range(NUM_EPISODES):
            s = env.reset()
            trajectory = []
            while True:
                probs = softmax(theta[s])
                a = sample(probs)
                s_next, r, done = env.step(a)
                trajectory.append((s, a, r, probs.copy()))
                s = s_next
                if done or len(trajectory) > 100: break

            # Compute returns
            T = len(trajectory)
            rets = np.zeros(T)
            G = 0.0
            for t in reversed(range(T)):
                G = trajectory[t][2] + GAMMA * G
                rets[t] = G

            # Compute gradient and Fisher matrix
            for t in range(T):
                s_t, a_t, _, probs = trajectory[t]
                adv = rets[t] - V[s_t]
                V[s_t] += 0.1 * (rets[t] - V[s_t])

                score = np.zeros(2)
                score[a_t] = 1.0
                score -= probs

                if method == "Vanilla":
                    theta[s_t] += ALPHA_VANILLA * score * adv
                else:
                    # ── Natural gradient ──
                    # Fisher matrix for this state: F = diag(π) − π·πᵀ
                    # For 2 actions, F is 2×2
                    F = np.diag(probs) - np.outer(probs, probs)
                    F += 1e-4 * np.eye(2)  # regularisation
                    F_inv = np.linalg.inv(F)

                    vanilla_grad = score * adv
                    natural_grad = F_inv @ vanilla_grad
                    theta[s_t] += ALPHA_NATURAL * natural_grad

            ep_reward = sum(r for _, _, r, _ in trajectory)
            rewards.append(ep_reward)

        results[method] = rewards
        avg = np.mean(rewards[-50:])
        print(f"  {method:>8s}: final avg reward = {avg:+.3f}")

    print(f"\n  Key insight: Natural gradient converges to the same solution")
    print(f"  but is invariant to how you parametrise the policy.")
    print(f"  With compatible function approximation: ∇_nat J(θ) = w")
    print(f"  (just use the critic weights as the gradient!)")


# ─────────────────────────────────────────────────────────────────────────────
# SIM 8: Summary Comparison
# ─────────────────────────────────────────────────────────────────────────────

def sim8_summary(rf_rewards, rf_grads, rfb_rewards, rfb_grads, ac_rewards, ac_grads):
    hdr("SIM 8: Summary — All Policy Gradient Variants")

    print(f"""
  From the slides, the policy gradient has many equivalent forms:

    ∇θ J(θ) = E[∇θ log π · vₜ]           REINFORCE
            = E[∇θ log π · Qw(s,a)]      Q Actor-Critic
            = E[∇θ log π · Aw(s,a)]      Advantage Actor-Critic
            = E[∇θ log π · δ]            TD Actor-Critic
            = E[∇θ log π · δ · e]        TD(λ) Actor-Critic
            G⁻¹ ∇θ J(θ) = w              Natural Actor-Critic

  Each replaces Q^π(s,a) with a different estimate:

  ┌──────────────────┬──────────────┬────────────┬───────────────────┐
  │ Method           │ Signal used  │ Bias       │ Variance          │
  ├──────────────────┼──────────────┼────────────┼───────────────────┤
  │ REINFORCE        │ Gₜ (return)  │ None       │ HIGH              │
  │ + Baseline       │ Gₜ − V(s)   │ None       │ Medium            │
  │ Actor-Critic     │ δ = r+γV'-V  │ Some       │ LOW               │
  │ A2C + Traces     │ δ · e        │ Some       │ Low, fast credit  │
  │ Natural AC       │ w (critic)   │ Some       │ Low + invariant   │
  └──────────────────┴──────────────┴────────────┴───────────────────┘
""")

    def avg_last(arr, n=50):
        return np.mean(arr[-n:]) if len(arr) >= n else np.mean(arr)

    print(f"  Performance comparison (avg reward, last 50 episodes):")
    print(f"    REINFORCE:          {avg_last(rf_rewards):+.3f}")
    print(f"    REINFORCE+Baseline: {avg_last(rfb_rewards):+.3f}")
    print(f"    Actor-Critic (TD):  {avg_last(ac_rewards):+.3f}")

    print(f"\n  Gradient variance comparison (avg |∇|, last 50 episodes):")
    print(f"    REINFORCE:          {avg_last(rf_grads):.4f}")
    print(f"    REINFORCE+Baseline: {avg_last(rfb_grads):.4f}  ← lower!")
    print(f"    Actor-Critic (TD):  {avg_last(ac_grads):.4f}  ← lowest!")

    print(f"\n  The progression: REINFORCE → +Baseline → Actor-Critic")
    print(f"  trades off BIAS for VARIANCE to get faster, more stable learning.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║  Policy Gradient Methods — David Silver Lecture 7                    ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                     ║
    ║  Core idea: directly parametrise the policy π_θ(s,a)                ║
    ║  and optimise θ by gradient ascent on J(θ).                         ║
    ║                                                                     ║
    ║  The Policy Gradient Theorem:                                       ║
    ║    ∇θ J(θ) = E_π [ ∇θ log π_θ(s,a) · Q^π(s,a) ]                  ║
    ║              ─────────────────────   ──────────                     ║
    ║              score function           how good                      ║
    ║              (which direction to       was this                     ║
    ║               push the policy)         action                      ║
    ║                                                                     ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    sim1_finite_difference()
    sim2_score_function()
    rf_r, rf_g = sim3_reinforce()
    rfb_r, rfb_g = sim4_reinforce_baseline()
    ac_r, ac_g = sim5_actor_critic()
    sim6_a2c_traces()
    sim7_natural_gradient()
    sim8_summary(rf_r, rf_g, rfb_r, rfb_g, ac_r, ac_g)

    print("\n" + "=" * 74)
    print("  ALL SIMULATIONS COMPLETE")
    print("=" * 74)
