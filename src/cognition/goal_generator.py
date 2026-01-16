import numpy as np
from collections import deque
import random

class GoalGenerator:
    def __init__(self, action_space):
        self.action_space = action_space
        self.active_goal = None
        self.goal_history = deque(maxlen=50)
        self.approach_threshold_pixels = 3000
        self.goal_timeout = 150
        self.goal_age = 0
        print("[COGNITION] GoalGenerator initialized (Information-Gain Driven)")

    def generate_goal(self, object_tracker, ipe):
        # generate next goal based on current knowledge state
        objects = object_tracker.get_all_objects()
        if not objects:
            return self._create_goal("explore_space", None, "forward")
        uncertain_options = []
        for obj_id in objects.keys():
            # TODO: for now, we only test causality with a forward push (Jetbot).
            uncertainty = ipe.get_uncertainty("forward", obj_id, object_tracker)
            if uncertainty > 0.5:
                uncertain_options.append(obj_id)
        if uncertain_options:
            target_id = random.choice(uncertain_options)
            target_obj = object_tracker.get_object(target_id)
            if target_obj:
                # if the object is too small/far, the goal is to approach it
                if target_obj["pixel_count"] < self.approach_threshold_pixels:
                    return self._create_goal("approach", target_id, None)
                # if the object is close enough, test causality with a push
                else:
                    return self._create_goal("test_causality", target_id, "forward")
        # if we are certain about everything, explore space to find new things
        return self._create_goal("explore_space", None, "forward")

    def _create_goal(self, goal_type, target_object_id, action):
        goal = {
            "type": goal_type,
            "target": target_object_id,
            "action": action,
            "created_at": len(self.goal_history)
        }
        self.goal_history.append(goal)
        self.active_goal = goal
        self.goal_age = 0
        return goal

    def get_active_goal(self):
        return self.active_goal

    def should_abandon_goal(self, object_tracker):
        if self.active_goal is None:
            return False
        self.goal_age += 1
        if self.goal_age > self.goal_timeout:
            return True
        if self.active_goal["type"] in ["approach", "test_causality"]:
            target_id = self.active_goal.get("target")
            if target_id is not None:
                obj = object_tracker.get_object(target_id)
                if obj is None:
                    return True
        return False

    def print_goal(self, goal):
        if goal is None:
            print("[GOAL] No active goal")
            return
        target_info = f"target={goal.get('target')}" if goal.get('target') is not None else "target=None"
        print(f"[GOAL] type={goal['type']} | {target_info} | action={goal.get('action')}")
