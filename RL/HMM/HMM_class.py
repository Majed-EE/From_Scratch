import numpy as np

class HMMModel:
    def __init__(self, channel_state=3, res_state=3, num_observations=5):
        self.channel_state = channel_state  # good(D3), medium(D2), bad(D1)
        self.res_state = res_state  # high (R3), medium (R2), low (R1)
        self.num_states = channel_state * res_state  # Total states
        self.num_observations = num_observations  # Delay observations

        # Initialize the model parameters
        self.pi = self._initialize_pi()
        self.pi_state = self._initialize_pi_state()
        self.A = self._initialize_transition_matrix()
        self.B = self._initialize_emission_matrix()
        self.observations = [f"L{i+1}" for i in range(num_observations)]
        self.states = [f"R{r+1}D{d+1}" for r in range(res_state) for d in range(channel_state)]

    def _initialize_pi(self):
        pi = np.random.rand(self.channel_state)
        return pi / pi.sum()

    def _initialize_pi_state(self):
        pi_state = np.zeros((self.num_states))
        pi_state[[0, 3, 6]] = self.pi[0] / self.res_state  # good
        pi_state[[1, 4, 7]] = self.pi[1] / self.res_state  # medium
        pi_state[[2, 5, 8]] = self.pi[2] / self.res_state  # bad
        return pi_state

    def _initialize_transition_matrix(self):
        P_high_res = np.random.rand(3, 3)
        P_mid_res = np.random.rand(3, 3)
        P_low_res = np.random.rand(3, 3)

        P_high_res = P_high_res / P_high_res.sum(axis=1, keepdims=True)
        P_mid_res = P_mid_res / P_mid_res.sum(axis=1, keepdims=True)
        P_low_res = P_low_res / P_low_res.sum(axis=1, keepdims=True)

        A = np.zeros((self.num_states, self.num_states))
        A[0:3, 0:3] = P_high_res
        A[3:6, 3:6] = P_mid_res
        A[6:, 6:] = P_low_res
        return A

    def _initialize_emission_matrix(self):
        B = np.random.rand(self.num_states, self.num_observations)
        return B / B.sum(axis=1, keepdims=True)

    def generate_sequence(self, length=10):
        """
        Generate a sequence from the HMM.

        Returns:
            obs_sequence  : list of observation symbols
            state_sequence: list of hidden states (indices)
        """
        num_states = self.A.shape[0]
        observations = self.B.shape[1]

        # Choose initial state
        current_state = np.random.choice(num_states, p=self.pi_state)

        state_sequence = [current_state]
        obs_sequence = []

        # Generate first observation
        obs = np.random.choice(observations, p=self.B[current_state])
        obs_sequence.append(obs)

        # Generate remaining sequence
        for _ in range(length - 1):
            # Transition to next state
            current_state = np.random.choice(num_states, p=self.A[current_state])
            state_sequence.append(current_state)

            # Emit observation
            obs = np.random.choice(observations, p=self.B[current_state])
            obs_sequence.append(obs)

        return obs_sequence, state_sequence

    def print_model_parameters(self):
        print("Initial State Distribution (pi):")
        print(self.pi_state)
        print("Sum:", self.pi_state.sum())

        print("\nTransition Matrix A (9x9):")
        print(self.A)
        print("Row sums:", self.A.sum(axis=1))

        print("\nEmission Matrix B (9x5):")
        print(self.B)
        print("Row sums:", self.B.sum(axis=1))

        print("\nObservation Symbols:")
        print(self.observations)

        print("\nStates:")
        print(self.states)