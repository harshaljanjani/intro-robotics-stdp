import numpy as np
from omni.isaac.core.objects import DynamicSphere, DynamicCuboid, DynamicCylinder, FixedCuboid

def setup_scene(world):
    world.scene.add_default_ground_plane()
    # scattered objects with varying properties
    objects = []
    # red balls - different sizes
    objects.append(world.scene.add(DynamicSphere(
        prim_path="/World/red_ball_small", name="red_ball_small",
        position=np.array([1.2, 0.3, 0.03]), radius=0.03, color=np.array([1.0, 0.0, 0.0])
    )))
    objects.append(world.scene.add(DynamicSphere(
        prim_path="/World/red_ball_large", name="red_ball_large",
        position=np.array([0.8, -0.5, 0.06]), radius=0.06, color=np.array([1.0, 0.0, 0.0])
    )))
    # blue cubes
    objects.append(world.scene.add(DynamicCuboid(
        prim_path="/World/blue_cube_1", name="blue_cube_1",
        position=np.array([-0.9, 0.7, 0.05]), scale=np.array([0.1, 0.1, 0.1]), color=np.array([0.0, 0.0, 1.0])
    )))
    objects.append(world.scene.add(DynamicCuboid(
        prim_path="/World/blue_cube_2", name="blue_cube_2",
        position=np.array([-1.2, -0.3, 0.05]), scale=np.array([0.1, 0.1, 0.1]), color=np.array([0.0, 0.0, 1.0])
    )))
    # green cylinders
    objects.append(world.scene.add(DynamicCylinder(
        prim_path="/World/green_cylinder_1", name="green_cylinder_1",
        position=np.array([0.5, 0.9, 0.08]), radius=0.04, height=0.15, color=np.array([0.0, 1.0, 0.0])
    )))
    objects.append(world.scene.add(DynamicCylinder(
        prim_path="/World/green_cylinder_2", name="green_cylinder_2",
        position=np.array([-0.6, -0.8, 0.08]), radius=0.04, height=0.15, color=np.array([0.0, 1.0, 0.0])
    )))
    # yellow spheres
    objects.append(world.scene.add(DynamicSphere(
        prim_path="/World/yellow_ball_1", name="yellow_ball_1",
        position=np.array([1.5, -0.9, 0.04]), radius=0.04, color=np.array([1.0, 1.0, 0.0])
    )))
    objects.append(world.scene.add(DynamicSphere(
        prim_path="/World/yellow_ball_2", name="yellow_ball_2",
        position=np.array([-1.5, 0.4, 0.04]), radius=0.04, color=np.array([1.0, 1.0, 0.0])
    )))
    # fixed obstacles - walls
    objects.append(world.scene.add(FixedCuboid(
        prim_path="/World/wall_north", name="wall_north",
        position=np.array([0.0, 2.0, 0.2]), scale=np.array([4.0, 0.1, 0.4]), color=np.array([0.3, 0.3, 0.3])
    )))
    objects.append(world.scene.add(FixedCuboid(
        prim_path="/World/wall_south", name="wall_south",
        position=np.array([0.0, -2.0, 0.2]), scale=np.array([4.0, 0.1, 0.4]), color=np.array([0.3, 0.3, 0.3])
    )))
    objects.append(world.scene.add(FixedCuboid(
        prim_path="/World/wall_east", name="wall_east",
        position=np.array([2.0, 0.0, 0.2]), scale=np.array([0.1, 4.0, 0.4]), color=np.array([0.3, 0.3, 0.3])
    )))
    objects.append(world.scene.add(FixedCuboid(
        prim_path="/World/wall_west", name="wall_west",
        position=np.array([-2.0, 0.0, 0.2]), scale=np.array([0.1, 4.0, 0.4]), color=np.array([0.3, 0.3, 0.3])
    )))
    return objects