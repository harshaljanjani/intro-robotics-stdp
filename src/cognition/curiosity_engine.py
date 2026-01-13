import numpy as np
import cupy as cp

class CuriosityEngine:
    def __init__(self, action_space, sensory_dim=128):
        self.action_space = action_space
        self.sensory_dim = sensory_dim
        self.prediction_window = 50
        self.sensory_history = []
        self.prediction_errors = []
        self.epsilon = 0.2
        self.action_history = []
        self.action_window = 10
        print("[COGNITION] v0 Curiosity Engine initialized (Prediction-Error Driven)")

    def _compute_sensory_summary(self, img_gpu):
        if img_gpu is None or img_gpu.shape[0] == 0:
            return cp.zeros(self.sensory_dim, dtype=cp.float32)
        # simple summary: downsample and flatten
        h, w, c = img_gpu.shape
        target_h, target_w = 8, 8
        step_h, step_w = max(1, h // target_h), max(1, w // target_w)
        downsampled = img_gpu[::step_h, ::step_w, :]
        flattened = downsampled.flatten()
        if flattened.shape[0] > self.sensory_dim:
            summary = flattened[:self.sensory_dim]
        else:
            summary = cp.zeros(self.sensory_dim, dtype=cp.float32)
            summary[:flattened.shape[0]] = flattened
        return summary

    def _predict_next_state(self):
        if len(self.sensory_history) < 2:
            return self.sensory_history[-1] if self.sensory_history else cp.zeros(self.sensory_dim, dtype=cp.float32)
        recent = self.sensory_history[-min(5, len(self.sensory_history)):]
        velocity = cp.mean(cp.array([recent[i+1] - recent[i] for i in range(len(recent)-1)]), axis=0)
        prediction = self.sensory_history[-1] + velocity
        return prediction

    def _compute_prediction_error(self, actual):
        if len(self.sensory_history) < 2:
            return 0.0
        predicted = self._predict_next_state()
        error = float(cp.linalg.norm(actual - predicted))
        return error

    def step(self, sensory_input, motor_rates):
        sensory_summary = self._compute_sensory_summary(sensory_input)
        pred_error = self._compute_prediction_error(sensory_summary)
        self.prediction_errors.append(pred_error)
        if len(self.prediction_errors) > self.prediction_window:
            self.prediction_errors.pop(0)
        self.sensory_history.append(sensory_summary)
        if len(self.sensory_history) > self.prediction_window:
            self.sensory_history.pop(0)
        # epsilon-greedy: explore vs exploit
        if np.random.rand() < self.epsilon:
            # explore: random action
            action_index = np.random.randint(0, len(self.action_space))
        else:
            # exploit: choose action that historically led to high prediction error
            if len(self.action_history) < self.action_window:
                action_index = np.random.randint(0, len(self.action_space))
            else:
                # NOTE: (open to better choices here) heuristic: cycle through actions to maximize novelty
                action_counts = {a: self.action_history[-self.action_window:].count(a) for a in range(len(self.action_space))}
                action_index = min(action_counts, key=action_counts.get)
        self.action_history.append(action_index)
        if len(self.action_history) > self.prediction_window:
            self.action_history.pop(0)
        return self.action_space[action_index]