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
parser.add_argument("--log_file", type=str, default="hopper_xml.log", help="File to log generated parameters.")
args = parser.parse_args()
disable_prob = args.disable_prob
range_len = args.range_len
index = args.index
output_dir = args.output_dir
log_file = args.log_file

# Randomly generate dimensions for different body parts
torso_base = 0.4
thigh_base = 0.45
leg_base = 0.5
lower_leg_base = 0.5
foot_base = 0.4
torso_length = max(0.001, round(torso_base + (random.gauss(0, 1) * torso_base * range_len), 3))
torso_half_length = torso_length / 2
thigh_length = max(0.001, round(thigh_base + (random.gauss(0, 1) * thigh_base * range_len), 3))
leg_length = max(0.001, round(leg_base + (random.gauss(0, 1) * leg_base * range_len), 3))
lower_leg_length = max(0.001, round(lower_leg_base + (random.gauss(0, 1) * lower_leg_base * range_len), 3))
foot_length = max(0.001, round(foot_base + (random.gauss(0, 1) * foot_base * range_len), 3))

# Create root element <mujoco>
root = ET.Element("mujoco", model="hopper")

# Include necessary files
ET.SubElement(root, "include", file="../../misc/skybox.xml")
ET.SubElement(root, "include", file="../../misc/visual.xml")
ET.SubElement(root, "include", file="../../misc/materials.xml")

# Add <compiler> element
ET.SubElement(root, "compiler", angle="degree", coordinate="global", inertiafromgeom="true")

# Add <default> element
default = ET.SubElement(root, "default")
ET.SubElement(default, "joint", armature="1", damping="1", limited="true")
ET.SubElement(default, "geom", conaffinity="1", condim="1", contype="1", margin="0.001", material="self", solimp=".8 .8 .01", solref=".02 1")
ET.SubElement(default, "motor", ctrllimited="true", ctrlrange="-.4 .4")

# Add <option> element
ET.SubElement(root, "option", integrator="RK4", timestep="0.002")

# Add <visual> element
visual = ET.SubElement(root, "visual")
ET.SubElement(visual, "map", znear="0.02")

# Add <worldbody> element
worldbody = ET.SubElement(root, "worldbody")

# Add <light> element
ET.SubElement(worldbody, "light", cutoff="100", diffuse="1 1 1", dir="-1 -0 -1.5", directional="true", exponent="1", pos="0 0 1.3", specular=".1 .1 .1")

# Add ground geometry
ET.SubElement(worldbody, "geom", conaffinity="1", condim="3", name="floor", pos="0 0 0", size="20 20 .125", type="plane", material="grid")

# Create torso body
torso = ET.SubElement(worldbody, "body", name="torso", pos=f"0 0 {torso_half_length + thigh_length + leg_length + lower_leg_length + 0.1}")
ET.SubElement(worldbody, "camera", name="tilted", mode="trackcom", pos="2.9 3.2 1.2", xyaxes="1 -.9 0 0 0 -1")
ET.SubElement(worldbody, "camera", name="track", mode="trackcom", pos="0 5 1.5", xyaxes="1 0 0 0 0 -1")
ET.SubElement(torso, "joint", armature="0", axis="1 0 0", damping="0", limited="false", name="rootx", pos="0 0 0", stiffness="0", type="slide")
ET.SubElement(torso, "joint", armature="0", axis="0 0 1", damping="0", limited="false", name="rootz", pos="0 0 0", ref=f"{torso_half_length + thigh_length + leg_length + lower_leg_length + 0.1}", stiffness="0", type="slide")
ET.SubElement(torso, "joint", armature="0", axis="0 1 0", damping="0", limited="false", name="rooty", pos=f"0 0 {torso_half_length + thigh_length + leg_length + lower_leg_length + 0.1}", stiffness="0", type="hinge")
ET.SubElement(torso, "geom", friction="0.9", fromto=f"0 0 {torso_length + thigh_length + leg_length + lower_leg_length + 0.1} 0 0 {thigh_length + leg_length + lower_leg_length + 0.1}", name="torso_geom", size="0.05", type="capsule")

# thigh element
thigh = ET.SubElement(torso, "body", name="thigh", pos=f"0 0 {thigh_length + leg_length + lower_leg_length + 0.1}")
ET.SubElement(thigh, "joint", axis="0 -1 0", name="thigh_joint", pos=f"0 0 {thigh_length + leg_length + lower_leg_length + 0.1}", range="-150 0", type="hinge")
ET.SubElement(thigh, "geom", friction="0.9", fromto=f"0 0 {thigh_length + leg_length + lower_leg_length + 0.1} 0 0 {leg_length + lower_leg_length + 0.1}", name="thigh_geom", size="0.05", type="capsule")

# leg element
leg = ET.SubElement(thigh, "body", name="leg", pos=f"0 0 {leg_length/2 + lower_leg_length + 0.1}")
ET.SubElement(leg, "joint", axis="0 -1 0", name="leg_joint", pos=f"0 0 {leg_length + lower_leg_length + 0.1}", range="-150 0", type="hinge")
ET.SubElement(leg, "geom", friction="0.9", fromto=f"0 0 {leg_length + lower_leg_length + 0.1} 0 0 {lower_leg_length + 0.1}", name="leg_geom", size="0.04", type="capsule")

# lower_leg element
lower_leg = ET.SubElement(leg, "body", name="lower_leg", pos=f"0 0 {lower_leg_length/2 + 0.1}")
ET.SubElement(lower_leg, "joint", axis="0 -1 0", name="lower_leg_joint", pos=f"0 0 {lower_leg_length + 0.1}", range="-150 0", type="hinge")
ET.SubElement(lower_leg, "geom", friction="0.9", fromto=f"0 0 {lower_leg_length + 0.1} 0 0 0.1", name="lower_leg_geom", size="0.04", type="capsule")

# foot element
foot = ET.SubElement(lower_leg, "body", name="foot", pos="foot_length/6 0 0.1")
ET.SubElement(foot, "joint", axis="0 -1 0", name="foot_joint", pos="0 0 0.1", range="-45 45", type="hinge")
ET.SubElement(foot, "geom", friction="2.0", fromto=f"{-foot_length/3} 0 0.1 {2*foot_length/3} 0 0.1", name="foot_geom", size="0.06", type="capsule")

# List joint & actuator.
joint_options = [("thigh_joint", 200), ("leg_joint", 200), ("lower_leg_joint", 200), ("foot_joint", 200)]
selected_joints = []
unselected_joints = []

# Random add actuator.
actuator = ET.SubElement(root, "actuator")
for joint_name, gear_value in joint_options:
    if random.random() > disable_prob:
        ET.SubElement(actuator, "motor", ctrllimited="true", ctrlrange="-1.0 1.0", gear=str(gear_value), joint=joint_name, name=joint_name)
        selected_joints.append(joint_name)
    else:
        unselected_joints.append(joint_name)

# Ensure at least one actuator is added.
if not selected_joints:
    joint_name, gear_value = random.choice(joint_options)
    ET.SubElement(actuator, "motor", ctrllimited="true", ctrlrange="-1.0 1.0", gear=str(gear_value), joint=joint_name, name=joint_name)
    selected_joints.append(joint_name)
    unselected_joints.remove(joint_name)

# Log the generated values
with open(log_file, "a") as log:
    log.write(f"hopper_5_{index}, range_len: {range_len}, disable_prob: {disable_prob}, "
              f"torso_length: {torso_length}, thigh_length: {thigh_length}, leg_length: {leg_length}, "
              f"lower_leg_length: {lower_leg_length}, foot_length: {foot_length}, "
              f"disabled_joints: {unselected_joints}\n")

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define output file path
output_file = os.path.join(output_dir, f"hopper_5_{index}.xml")

# Format and write XML file
xml_str = minidom.parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="  ")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(xml_str)

print(f"XML file generated and saved to {output_file}")
