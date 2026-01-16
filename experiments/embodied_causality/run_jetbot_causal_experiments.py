import sys
from pathlib import Path
import numpy as np
import cupy as cp
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from isaacsim import SimulationApp
simulation_app_instance = SimulationApp({"headless": False})
from omni.isaac.core import World
from src.environment import rich_playground_scene
from src.agent.jetbot_controller import JetbotController
from src.perception.vision import VisionSystem
from src.perception.touch import TouchSystem
from src.perception.proprioception import ProprioceptionSystem
from src.perception import spike_encoder
from src.cognition.object_tracker import ObjectTracker
from src.cognition.causal_hypothesis_generator import CausalHypothesisGenerator
from src.cognition.goal_generator import GoalGenerator
from src.cognition.curiosity_engine import CuriosityEngine
from src.utils.loader import load_config
from src.network.builder import build_network
from src.engine.simulate import Simulator
from src.utils.causal_analysis_utils import (
    track_causal_graph_snapshot,
    track_goal_statistics,
    save_causal_learning_plots,
    compute_average_uncertainty,
    print_causal_summary
)

def run_simulation():
    base_path = Path(__file__).resolve().parent.parent.parent
    network_config = load_config(base_path / "config" / "network" / "embodied_jetbot_v0.json")
    sim_config = load_config(base_path / "config" / "simulation" / "default_run.json")
    world = World(stage_units_in_meters=1.0)
    rich_playground_scene.setup_scene(world)
    robot_controller = JetbotController()
    robot = robot_controller.create_robot(world)
    world.reset()
    robot_controller.initialize()
    robot_controller.forward_velocity = 30.0
    robot_controller.turn_velocity = 20.0
    camera_path = "/World/JetbotCamera"
    vision = VisionSystem(
        camera_prim_path=camera_path,
        attachment_prim_path="/World/Jetbot/chassis",
        offset_position=[0.1, 0.0, 0.01]
    )
    vision.initialize(world)
    touch = TouchSystem(robot_prim_path="/World/Jetbot")
    touch.initialize()
    proprio = ProprioceptionSystem(robot_articulation=robot)
    proprio.initialize()
    network, pop_info = build_network(network_config)
    snn_simulator = Simulator(network, pop_info, sim_config)
    object_tracker = ObjectTracker()
    chg = CausalHypothesisGenerator(network, pop_info, confidence_threshold=0.3)
    action_space = ["forward", "turn_left", "turn_right", "stop"]
    curiosity_engine = CuriosityEngine(action_space)
    goal_generator = GoalGenerator(action_space)
    vision_pop_map = {
        "Vision_Left": "Vision_Left",
        "Vision_Center": "Vision_Center",
        "Vision_Right": "Vision_Right"
    }
    target_color_tensor = cp.array([1.0, 0.0, 0.0], dtype=cp.float32)
    # tracking structures
    causal_history = {
        'sample_steps': [],
        'motor_to_object_strengths': {
            'Motor_Forward': [],
            'Motor_Turn_L': [],
            'Motor_Turn_R': []
        },
        'graph_metrics': [],
        'goal_statistics': [],
        'uncertainty_progression': [],
        'unique_objects_over_time': []
    }
    unique_objects_seen = set()
    print("\n=== STARTING CAUSAL EXPERIMENTS ===")
    max_steps = 3000
    report_interval = 500
    sample_interval = 100
    goal_duration = 80
    steps_in_current_goal = 0
    active_goal = None
    for step in range(max_steps):
        current_time_ms = step * sim_config["dt"]
        vision.update_camera_pose()
        rgba_data = vision.camera.get_rgba()
        if rgba_data is None or rgba_data.shape[0] == 0:
            world.step(render=True)
            continue
        img_gpu = cp.asarray(rgba_data[..., :3])
        object_tracker.update_from_vision(img_gpu, current_time_ms)
        # track unique objects
        for obj_id in object_tracker.get_all_objects().keys():
            unique_objects_seen.add(obj_id)
        # sample causal graph state
        if step > 0 and step % sample_interval == 0:
            chg.build_causal_graph()
            causal_history['sample_steps'].append(step)
            # track motor → object causality strengths
            for motor_name in ['Motor_Forward', 'Motor_Turn_L', 'Motor_Turn_R']:
                strength = chg.causal_chains.get(motor_name, {}).get("Targeted_Object_Motion", 0.0)
                causal_history['motor_to_object_strengths'][motor_name].append(strength)
            # track graph size
            snapshot = track_causal_graph_snapshot(chg, step)
            causal_history['graph_metrics'].append(snapshot)
            # track uncertainty
            avg_uncertainty = compute_average_uncertainty(chg, object_tracker, action_space)
            causal_history['uncertainty_progression'].append(avg_uncertainty)
            # track unique objects
            causal_history['unique_objects_over_time'].append(len(unique_objects_seen))
        if step > 0 and step % report_interval == 0:
            print(f"\n=== Step {step}/{max_steps} ===")
            chg.build_causal_graph()
            chg.print_causal_graph()
        robot_pos, _ = robot.get_world_pose()
        # decision hierarchy: recovery > goals > curiosity
        # 1. check recovery (highest priority)
        curiosity_engine.recovery_detector.update(robot_pos, curiosity_engine.action_history[-1] if curiosity_engine.action_history else "stop")
        if curiosity_engine.recovery_mode:
            action = curiosity_engine.recovery_sequence[curiosity_engine.recovery_steps]
            curiosity_engine.recovery_steps += 1
            if curiosity_engine.recovery_steps >= len(curiosity_engine.recovery_sequence):
                curiosity_engine.recovery_mode = False
                curiosity_engine.recovery_steps = 0
                curiosity_engine.recovery_cooldown = curiosity_engine.recovery_cooldown_duration
                print("[CURIOSITY] Recovery complete, cooldown active")
            curiosity_engine.action_history.append(action)
            steps_in_current_goal = goal_duration
        elif curiosity_engine.recovery_detector.is_stuck() and curiosity_engine.recovery_cooldown == 0:
            curiosity_engine.recovery_mode = True
            curiosity_engine.recovery_steps = 0
            print("[CURIOSITY] WALL DETECTED! Executing recovery sequence.")
            action = curiosity_engine.recovery_sequence[0]
            curiosity_engine.action_history.append(action)
            steps_in_current_goal = goal_duration
        else:
            # 2. goal-directed behavior (when not recovering)
            if curiosity_engine.recovery_cooldown > 0:
                curiosity_engine.recovery_cooldown -= 1
            if steps_in_current_goal >= goal_duration or active_goal is None:
                robot_controller.stop()
                active_goal = goal_generator.generate_goal(object_tracker, chg)
                goal_generator.print_goal(active_goal)
                # track goal
                goal_stat = track_goal_statistics(active_goal, step)
                causal_history['goal_statistics'].append(goal_stat)
                steps_in_current_goal = 0
            # execute the current goal's logic for this step
            if active_goal is not None:
                action_from_goal = active_goal.get("action")
                goal_type = active_goal.get("type")
                target_id = active_goal.get("target")
                if goal_type == "approach":
                    target_obj = object_tracker.get_object(target_id)
                    if target_obj:
                        obj_x_pos = target_obj["position"][0]
                        if obj_x_pos < 0.4:
                            action = "turn_left"
                        elif obj_x_pos > 0.6:
                            action = "turn_right"
                        else:
                            action = "forward"
                    else:
                        # target lost, force new goal
                        active_goal = None
                        action = "forward"
                elif action_from_goal:
                    action = action_from_goal
                else:
                    action = "forward"
            else:
                action = "forward"
            curiosity_engine.action_history.append(action)
        if action == "forward":
            robot_controller.forward()
        elif action == "turn_left":
            robot_controller.turn_left()
        elif action == "turn_right":
            robot_controller.turn_right()
        else:
            robot_controller.stop()
        # learning phase; independent of action
        target_motion_spikes = None
        motor_command_spikes = None
        if active_goal and active_goal["type"] == "test_causality":
            target_id = active_goal.get("target")
            if target_id is not None:
                obj = object_tracker.get_object(target_id)
                if obj is not None:
                    vel_mag = np.linalg.norm(obj["velocity"])
                    target_motion_spikes = proprio.encode_motion_to_spikes(
                        pop_info, "Targeted_Object_Motion", vel_mag, threshold=0.01, sensitivity=5.0
                    )
            if action == "forward":
                pop = pop_info['Motor_Forward']
                motor_command_spikes = cp.arange(pop['start'], pop['end'], dtype=cp.int32)
            elif action == "turn_left":
                pop = pop_info['Motor_Turn_L']
                motor_command_spikes = cp.arange(pop['start'], pop['end'], dtype=cp.int32)
            elif action == "turn_right":
                pop = pop_info['Motor_Turn_R']
                motor_command_spikes = cp.arange(pop['start'], pop['end'], dtype=cp.int32)
        vision_spikes = spike_encoder.encode_spatial_location(img_gpu, pop_info, vision_pop_map, target_color_tensor)
        contact_count = touch.get_contact_count()
        touch_spikes = touch.encode_touch_to_spikes(pop_info, "Touch_Sensor", contact_count)
        motion_intensity = proprio.get_motion_intensity()
        proprio_spikes = proprio.encode_motion_to_spikes(pop_info, "Proprio_Motion", motion_intensity)
        spike_lists = [s for s in [vision_spikes, touch_spikes, proprio_spikes, target_motion_spikes, motor_command_spikes] if s is not None]
        sensory_spikes = cp.concatenate(spike_lists) if spike_lists else None
        snn_simulator.step(sensory_spikes)
        world.step(render=True)
        steps_in_current_goal += 1
    print("\n=== CAUSAL EXPERIMENTS COMPLETE ===")
    chg.build_causal_graph()
    chg.print_causal_graph()
    # print summary
    print_causal_summary(causal_history)
    # save visualizations
    output_dir = base_path / "results" / "visualizations" / "embodied_causality"
    save_causal_learning_plots(causal_history, output_dir, experiment_name="jetbot_causal_learning")
    simulation_app_instance.close()

if __name__ == "__main__":
    run_simulation()