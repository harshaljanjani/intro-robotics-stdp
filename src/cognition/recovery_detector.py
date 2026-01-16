import numpy as np
from collections import deque

class RecoveryDetector:
    # motion-based stuck detection - works well, no need for learned policy
    # the snn itself should learn obstacle avoidance via causal learning
    def __init__(self, stuck_duration=40, forward_threshold=0.05, turn_threshold=0.02, action_check_window=15):
        self.stuck_duration = stuck_duration
        self.forward_threshold = forward_threshold
        self.turn_threshold = turn_threshold
        self.action_check_window = action_check_window
        self.position_history = deque(maxlen=stuck_duration)
        self.action_history = deque(maxlen=action_check_window)
        self.was_stuck_last_check = False
        print(f"[COGNITION] RecoveryDetector initialized (Motion-Based): duration={stuck_duration}, forward_thresh={forward_threshold}m, turn_thresh={turn_threshold}m")

    def update(self, robot_position, action):
        if robot_position is not None:
            self.position_history.append(robot_position[:2]) 
        self.action_history.append(action)

    def is_stuck(self):
        if len(self.position_history) < self.stuck_duration:
            return False
        was_trying_to_move_forward = "forward" in self.action_history
        if not was_trying_to_move_forward:
            return False
        start_position = self.position_history[0]
        current_position = self.position_history[-1]
        distance_moved = np.linalg.norm(current_position - start_position)
        forward_count = list(self.action_history).count("forward")
        turn_count = list(self.action_history).count("turn_left") + list(self.action_history).count("turn_right")
        if forward_count > turn_count:
            threshold = self.forward_threshold
        else:
            threshold = self.turn_threshold
        is_stuck = distance_moved < threshold
        if is_stuck and not self.was_stuck_last_check:
            print(f"[RECOVERY_DETECTOR] Stuck detected! Moved only {distance_moved:.4f}m in last {self.stuck_duration} steps (threshold: {threshold:.4f}m).")
        self.was_stuck_last_check = is_stuck
        return is_stuck
