import os
import argparse
import math
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate a MuJoCo XML file with random dimensions.")
parser.add_argument("--disable_prob", type=float, default=0.0, help="Probability of disabling a joint actuator.")
parser.add_argument("--range_len", type=float, default=0.1, help="Range length for body part length variation.")
parser.add_argument("--index", type=int, help="The model number (integer) for the generated XML file.")
parser.add_argument("--output_dir", type=str, default="../xmls/", help="Directory to save the generated XML file.")
parser.add_argument("--log_file", type=str, default="humanoid_xml.log", help="File to log generated parameters.")
args = parser.parse_args()
disable_prob = args.disable_prob
range_len = args.range_len
index = args.index
output_dir = args.output_dir
log_file = args.log_file

# Randomly generate dimensions for different body parts
left_thigh_base = 0.34
right_thigh_base = 0.34
left_shin_base = 0.3
right_shin_base = 0.3
right_upper_arm_base = 0.277
left_upper_arm_base = 0.277
right_lower_arm_base = 0.294
left_lower_arm_base = 0.294
left_thigh_length = max(0.001, round(left_thigh_base + (random.gauss(0, 1) * left_thigh_base * range_len), 3))
right_thigh_length = max(0.001, round(right_thigh_base + (random.gauss(0, 1) * right_thigh_base * range_len), 3))
# left_shin_length = max(0.001, round(left_shin_base + (random.gauss(0, 1) * left_shin_base * range_len), 3))
right_shin_length = max(0.001, round(right_shin_base + (random.gauss(0, 1) * right_shin_base * range_len), 3))
right_upper_arm_length = max(0.001, round(right_upper_arm_base + (random.gauss(0, 1) * right_upper_arm_base * range_len), 3))
left_upper_arm_length = max(0.001, round(left_upper_arm_base + (random.gauss(0, 1) * left_upper_arm_base * range_len), 3))
right_lower_arm_length = max(0.001, round(right_lower_arm_base + (random.gauss(0, 1) * right_lower_arm_base * range_len), 3))
left_lower_arm_length = max(0.001, round(left_lower_arm_base + (random.gauss(0, 1) * left_lower_arm_base * range_len), 3))
left_leg_2_torso_height = 0.063 + left_thigh_length + 0.465
right_leg_2_torso_height = 0.1 + right_shin_length + 0.063 + right_thigh_length + 0.465
torso_height = max(left_leg_2_torso_height, right_leg_2_torso_height)

# Create root element <mujoco>
root = ET.Element("mujoco", model="humanoid")

# Include necessary files
ET.SubElement(root, "include", file="../../misc/skybox.xml")
ET.SubElement(root, "include", file="../../misc/visual.xml")
ET.SubElement(root, "include", file="../../misc/materials.xml")

# Add <compiler> element
ET.SubElement(root, "compiler", angle="degree", inertiafromgeom="true")

# Add <default> element
default = ET.SubElement(root, "default")
ET.SubElement(default, "joint", armature="1", damping="1", limited="true")
ET.SubElement(default, "geom", conaffinity="0", condim="3", contype="1", density="1000", friction=".7 .1 .1", material="self")
ET.SubElement(default, "motor", ctrllimited="true", ctrlrange="-1 1")

# Add <option> element
ET.SubElement(root, "option", integrator="RK4", timestep="0.002")

# Add <visual> element
visual = ET.SubElement(root, "visual")
ET.SubElement(visual, "map", fogend="5", fogstart="3")

# Add <worldbody> element
worldbody = ET.SubElement(root, "worldbody")

# Add <light> element
ET.SubElement(worldbody, "light", cutoff="100", diffuse="1 1 1", dir="-1 -0 -1.5", directional="true", exponent="1", pos="0 0 1.3", specular=".1 .1 .1")

# Add ground geometry
ET.SubElement(worldbody, "geom", conaffinity="1", condim="3", material="grid", name="floor", pos="0 0 0", size="20 20 0.125", type="plane")

# Add torso
torso = ET.SubElement(worldbody, "body", name="torso", pos=f"0 0 {torso_height + 0.2}")

# Add joint
ET.SubElement(torso, "joint", armature="0.02", axis="1 0 0", damping="0", name="abdomen_x", pos="0 0 0", stiffness="0", type="slide", limited="false")
ET.SubElement(torso, "joint", armature="0.02", axis="0 0 1", damping="0", name="abdomen_z", pos="0 0 0", ref="1.25", stiffness="0", type="slide", limited="false")
ET.SubElement(torso, "joint", armature="0.02", axis="0 1 0", damping="0", name="abdomen_y", pos="0 0 {torso_height - 0.3}", stiffness="0", type="hinge", limited="false")

# Add torso
ET.SubElement(torso, "geom", fromto="0 -.07 0 0 .07 0", name="torso1", size="0.07", type="capsule")

# Add head
ET.SubElement(torso, "geom", name="head", pos="0 0 .19", size=".09", type="sphere")

# Add uwaist
ET.SubElement(torso, "geom", fromto="-.01 -.06 -.12 -.01 .06 -.12", name="uwaist", size="0.06", type="capsule")

# Add lwaist
ET.SubElement(torso, "geom", fromto="-.01 -.06 -0.260 -.01 .06 -0.260", name="lwaist", size="0.06", quat="1.000 0 -0.002 0", type="capsule")

# Add butt
ET.SubElement(torso, "geom", fromto="-.02 -.07 -0.425 -.02 .07 -0.425", name="butt", size="0.09", quat="1.000 0 -0.002 0", type="capsule")

# Add right thigh
right_thigh = ET.SubElement(torso, "body", name="right_thigh", pos="0 -0.1 -0.465")
ET.SubElement(right_thigh, "joint", armature="0.0080", axis="0 1 0", damping="5", name="right_hip_y", pos="0 0 0", range="-110 20", stiffness="20", type="hinge")
ET.SubElement(right_thigh, "geom", fromto=f"0 0 0 0 0.01 {-right_thigh_length}", name="right_thigh1", size="0.06", type="capsule")

# Add right shin
right_shin = ET.SubElement(right_thigh, "body", name="right_shin", pos=f"0 0.01 {-(right_thigh_length + 0.063)}")
ET.SubElement(right_shin, "joint", armature="0.0060", axis="0 -1 0", name="right_knee", pos="0 0 .02", range="-160 -2", type="hinge")
ET.SubElement(right_shin, "geom", fromto=f"0 0 0 0 0 {-right_shin_length}", name="right_shin1", size=f"0.049", type="capsule")
ET.SubElement(right_shin, "geom", name="right_foot", pos=f"0 0 {-right_shin_length - 0.05}", size="0.075", type="sphere")

# Add left thigh
left_thigh = ET.SubElement(torso, "body", name="left_thigh", pos="0 0.1 -0.465")
ET.SubElement(left_thigh, "joint", armature="0.01", axis="0 1 0", damping="5", name="left_hip_y", pos="0 0 0", range="-110 20", stiffness="20", type="hinge")
ET.SubElement(left_thigh, "geom", fromto=f"0 0 0 0 -0.01 {-left_thigh_length}", name="left_thigh1", size=f"0.06", type="capsule")

# Generate left shin
# left_shin = ET.SubElement(left_thigh, "body", name="left_shin", pos=f"0 -0.01 {-(left_thigh_length + 0.063)}")
# ET.SubElement(left_shin, "joint", armature="0.0060", axis="0 -1 0", name="left_knee", pos="0 0 .02", range="-160 -2", stiffness="1", type="hinge")
# ET.SubElement(left_shin, "geom", fromto=f"0 0 0 0 0 {-left_shin_length}", name="left_shin1", size="0.049", type="capsule")
# ET.SubElement(left_shin, "geom", name="left_foot", type="sphere", size="0.075", pos=f"0 0 {-left_shin_length - 0.05}")

# Add right upper arm
right_upper_arm = ET.SubElement(torso, "body", name="right_upper_arm", pos="0 -0.17 0.06")
ET.SubElement(right_upper_arm, "joint", armature="0.0060", axis="0 -1 0", name="right_shoulder1", pos="0 0 0", range="-85 60", stiffness="1", type="hinge")
ET.SubElement(right_upper_arm, "geom", fromto=f"0 0 0 {right_upper_arm_length / (3 ** 0.5)} {-right_upper_arm_length / (3 ** 0.5)} {-right_upper_arm_length / (3 ** 0.5)}", name="right_uarm1", size="0.04 0.16", type="capsule")

# Add right lower arm
right_lower_arm = ET.SubElement(right_upper_arm, "body", name="right_lower_arm", pos=f"{0.02 + right_upper_arm_length / (3 ** 0.5)} {-(0.02 + right_upper_arm_length / (3 ** 0.5))} {-(0.02 + right_upper_arm_length / (3 ** 0.5))}")
ET.SubElement(right_lower_arm, "joint", armature="0.0028", axis="0 -1 0", name="right_elbow", pos="0 0 0", range="-90 50", stiffness="0", type="hinge")
ET.SubElement(right_lower_arm, "geom", fromto=f"0.01 0.01 0.01 {right_upper_arm_length / (3 ** 0.5)} {right_upper_arm_length / (3 ** 0.5)} {right_upper_arm_length / (3 ** 0.5)}", name="right_larm", size="0.031", type="capsule")
ET.SubElement(right_lower_arm, "geom", name="right_hand", pos=f"{right_upper_arm_length / (3 ** 0.5) + 0.01} {right_upper_arm_length / (3 ** 0.5) + 0.01} {right_upper_arm_length / (3 ** 0.5) + 0.01}", size="0.04", type="sphere")

# Add left upper arm
left_upper_arm = ET.SubElement(torso, "body", name="left_upper_arm", pos="0 0.17 0.06")
ET.SubElement(left_upper_arm, "joint", armature="0.0060", axis="0 -1 0", name="left_shoulder1", pos="0 0 0", range="-60 85", stiffness="1", type="hinge")
ET.SubElement(left_upper_arm, "geom", fromto=f"0 0 0 {left_upper_arm_length / (3 ** 0.5)} {left_upper_arm_length / (3 ** 0.5)} {-left_upper_arm_length / (3 ** 0.5)}", name="left_uarm1", size="0.04 0.16", type="capsule")

# Add left lower arm
left_lower_arm = ET.SubElement(left_upper_arm, "body", name="left_lower_arm", pos=f"{0.02 + left_upper_arm_length / (3 ** 0.5)} {0.02 + left_upper_arm_length / (3 ** 0.5)} {-(0.02 + left_upper_arm_length / (3 ** 0.5))}")
ET.SubElement(left_lower_arm, "joint", armature="0.0040", axis="0 -1 0", name="left_elbow", pos="0 0 0", range="-90 50", stiffness="0", type="hinge")
ET.SubElement(left_lower_arm, "geom", fromto=f"0.01 -0.01 0.01 {right_upper_arm_length / (3 ** 0.5)} {-right_upper_arm_length / (3 ** 0.5)} {right_upper_arm_length / (3 ** 0.5)}", name="left_larm", size="0.031", type="capsule")
ET.SubElement(left_lower_arm, "geom", name="left_hand", pos=f"{right_upper_arm_length / (3 ** 0.5) + 0.01} {-right_upper_arm_length / (3 ** 0.5) - 0.01} {right_upper_arm_length / (3 ** 0.5) + 0.01}", size="0.04", type="sphere")

# Add camera
ET.SubElement(torso, "camera", mode="trackcom", name="track", pos="0 6 0", xyaxes="1 0 0 0 0 -1")
ET.SubElement(torso, "camera", mode="trackcom", name="tilted", pos="2.7 3 0", xyaxes="1 -.9 0 0 0 -1")

# List joint & actuator.
joint_options = [("right_hip_y", 150), ("right_knee", 100), ("left_hip_y", 150), ("right_shoulder1", 12.5), 
                 ("right_elbow", 12.5), ("left_shoulder1", 12.5), ("left_elbow", 12.5)]
selected_joints = []
unselected_joints = []

# Random add actuator.
actuator = ET.SubElement(root, "actuator")
for joint_name, gear_value in joint_options:
    if random.random() > disable_prob:
        ET.SubElement(actuator, "motor", gear=str(gear_value), joint=joint_name, name=joint_name)
        selected_joints.append(joint_name)
    else:
        unselected_joints.append(joint_name)

# Ensure at least one actuator is added.
if not selected_joints:
    joint_name, gear_value = random.choice(joint_options)
    ET.SubElement(actuator, "motor", gear=str(gear_value), joint=joint_name, name=joint_name)
    selected_joints.append(joint_name)
    unselected_joints.remove(joint_name)

# Log the generated values
with open(log_file, "a") as log:
    log.write(f"humanoid_2d_8_left_knee_{index}, range_len: {range_len}, disable_prob: {disable_prob}, "
              f"left_thigh_length: {left_thigh_length}, right_thigh_length: {right_thigh_length}, "
              f"left_shin_length: disabled, right_shin_length: {right_shin_length}, "
              f"left_upper_arm_length: {left_upper_arm_length}, right_upper_arm_length: {right_upper_arm_length}, "
              f"left_lower_arm_length: {left_lower_arm_length}, right_lower_arm_length: {right_lower_arm_length}, "
              f"disabled_joints: {unselected_joints}\n")

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define output file path
output_file = os.path.join(output_dir, f"humanoid_2d_8_left_knee_{index}.xml")

# Format and write XML file
xml_str = minidom.parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="  ")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(xml_str)

print(f"XML file generated and saved to {output_file}")
