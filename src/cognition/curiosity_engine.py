import numpy as np
import cupy as cp
from collections import deque
from src.cognition.recovery_detector import RecoveryDetector

class CuriosityEngine:
    def __init__(self, action_space, sensory_dim=128):
        self.action_space = action_space
        self.sensory_dim = sensory_dim
        self.prediction_window = 50
        self.sensory_history = deque(maxlen=self.prediction_window)
        self.prediction_errors = deque(maxlen=self.prediction_window)
        self.action_history = deque(maxlen=20)
        self.recovery_detector = RecoveryDetector()
        self.recovery_mode = False
        self.recovery_steps = 0
        self.recovery_sequence = ["turn_right"] * 8 + ["forward"] * 3
        self.recovery_cooldown = 0
        self.recovery_cooldown_duration = 200
        self.action_novelty_scores = {action: deque(maxlen=10) for action in action_space}
        self.forward_bias = 0.6
        print("[COGNITION] v0 Curiosity Engine initialized (Novelty-Seeking + Anti-Perseveration)")

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

    def _compute_novelty(self, current_summary):
        if len(self.sensory_history) < 5:
            return 1.0
        recent = list(self.sensory_history)[-5:]
        min_distance = float('inf')
        for past_state in recent:
            distance = float(cp.linalg.norm(current_summary - past_state))
            min_distance = min(min_distance, distance)
        novelty = min_distance / 100.0
        return min(1.0, novelty)

    def _compute_action_diversity_score(self):
        if len(self.action_history) < 5:
            return 1.0
        recent_actions = list(self.action_history)[-5:]
        unique_actions = len(set(recent_actions))
        diversity = unique_actions / 5.0
        return diversity

    def step(self, sensory_input, robot_position):
        sensory_summary = self._compute_sensory_summary(sensory_input)
        novelty = self._compute_novelty(sensory_summary)
        self.sensory_history.append(sensory_summary)
        current_action = self.action_history[-1] if self.action_history else "stop"
        self.recovery_detector.update(robot_position, current_action)
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1
        # if in recovery mode, continue the sequence
        if self.recovery_mode:
            action = self.recovery_sequence[self.recovery_steps]
            self.recovery_steps += 1
            if self.recovery_steps >= len(self.recovery_sequence):
                self.recovery_mode = False
                self.recovery_steps = 0
                self.recovery_cooldown = self.recovery_cooldown_duration
                print("[CURIOSITY] Recovery complete, cooldown active")
            self.action_history.append(action)
            return action # explicitly return the recovery action
        # check if we should enter recovery mode
        if self.recovery_detector.is_stuck() and self.recovery_cooldown == 0:
            self.recovery_mode = True
            self.recovery_steps = 0
            print("[CURIOSITY] WALL DETECTED! Executing recovery sequence.")
            action = self.recovery_sequence[0] # the first recovery action
            self.action_history.append(action)
            return action
        # priority 3: novelty-seeking with anti-perseveration
        if len(self.action_history) >= 3:
            last_action = self.action_history[-1]
            self.action_novelty_scores[last_action].append(novelty)
        diversity = self._compute_action_diversity_score()
        recent_action_counts = {}
        recent = list(self.action_history)[-10:] if len(self.action_history) >= 10 else list(self.action_history)
        for act in self.action_space:
            recent_action_counts[act] = recent.count(act)
        action_scores = {}
        for act in self.action_space:
            avg_novelty = np.mean(list(self.action_novelty_scores[act])) if len(self.action_novelty_scores[act]) > 0 else 0.5
            recency_penalty = recent_action_counts[act] / max(1, len(recent))
            forward_bonus = self.forward_bias if act == "forward" else 0.0
            action_scores[act] = avg_novelty - (recency_penalty * 0.5) + forward_bonus
        if diversity < 0.4:
            least_used = min(recent_action_counts, key=recent_action_counts.get)
            action = least_used
        else:
            action = max(action_scores.keys(), key=lambda k: action_scores[k])
        self.action_history.append(action)
        return action

    def get_average_novelty(self):
        if len(self.sensory_history) < 2:
            return 0.0
        recent_novelties = []
        recent = list(self.sensory_history)[-10:]
        for i in range(1, len(recent)):
            novelty = self._compute_novelty(recent[i])
            recent_novelties.append(novelty)
        return np.mean(recent_novelties) if recent_novelties else 0.0

    def get_most_surprising_object(self, object_tracker):
        objects = object_tracker.get_all_objects()
        if not objects:
            return None
        max_surprise = 0.0
        most_surprising = None
        for obj_id, obj_data in objects.items():
            vel_magnitude = np.linalg.norm(obj_data["velocity"])
            if vel_magnitude > max_surprise:
                max_surprise = vel_magnitude
                most_surprising = obj_id
        return most_surprising