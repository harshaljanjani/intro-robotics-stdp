import numpy as np
import cupy as cp

class ProprioceptionSystem:
    def __init__(self, robot_articulation):
        self.robot = robot_articulation
        self.velocity_history = []
        self.history_length = 10

    def initialize(self):
        print("[PROPRIO] Proprioception system initialized")

    def get_joint_velocities(self):
        if self.robot and hasattr(self.robot, 'get_joint_velocities'):
            velocities = self.robot.get_joint_velocities()
            return velocities
        return None

    def get_motion_intensity(self):
        velocities = self.get_joint_velocities()
        if velocities is None:
            return 0.0
        motion = np.linalg.norm(velocities)
        self.velocity_history.append(motion)
        if len(self.velocity_history) > self.history_length:
            self.velocity_history.pop(0)
        return np.mean(self.velocity_history)

    def encode_motion_to_spikes(self, pop_info, pop_name, motion_intensity, threshold=0.1, sensitivity=1.0):
        if motion_intensity < threshold:
            return None
        pop = pop_info[pop_name]
        num_firing = int(pop['count'] * min(1.0, (motion_intensity / 10.0) * sensitivity))
        if num_firing == 0:
            return None
        indices = cp.random.choice(
            cp.arange(pop['start'], pop['end']),
            size=num_firing,
            replace=False
        )
        return indices.astype(cp.int32)
