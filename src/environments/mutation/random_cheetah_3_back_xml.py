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
parser.add_argument("--log_file", type=str, default="cheetah_xml.log", help="File to log generated parameters.")
args = parser.parse_args()
disable_prob = args.disable_prob
range_len = args.range_len
index = args.index
output_dir = args.output_dir
log_file = args.log_file

# Randomly generate dimensions for different body parts
torso_base = 1.0
bthigh_base = 0.145
bshin_base = 0.15
bfoot_base = 0.094
fthigh_base = 0.133
fshin_base = 0.106
ffoot_base = 0.07
torso_length = max(0.001, round(torso_base + (random.gauss(0, 1) * torso_base * range_len), 3))
torso_half_length = torso_length / 2
bthigh_length = max(0.001, round(bthigh_base + (random.gauss(0, 1) * bthigh_base * range_len), 3))
bshin_length = max(0.001, round(bshin_base + (random.gauss(0, 1) * bshin_base * range_len), 3))
bfoot_length = max(0.001, round(bfoot_base + (random.gauss(0, 1) * bfoot_base * range_len), 3))
fthigh_length = max(0.001, round(fthigh_base + (random.gauss(0, 1) * fthigh_base * range_len), 3))
fshin_length = max(0.001, round(fshin_base + (random.gauss(0, 1) * fshin_base * range_len), 3))
ffoot_length = max(0.001, round(ffoot_base + (random.gauss(0, 1) * ffoot_base * range_len), 3))

# Create root element <mujoco>
root = ET.Element("mujoco", model="cheetah")

# Include necessary files
ET.SubElement(root, "include", file="../../misc/skybox.xml")
ET.SubElement(root, "include", file="../../misc/visual.xml")
ET.SubElement(root, "include", file="../../misc/materials.xml")

# Add <compiler> element
ET.SubElement(root, "compiler", angle="radian", coordinate="local", inertiafromgeom="true")

# Add <default> element
default = ET.SubElement(root, "default")
ET.SubElement(default, "joint", armature=".1", damping=".01", limited="true", solimplimit="0 .8 .03", solreflimit=".02 1", stiffness="8")
ET.SubElement(default, "geom", conaffinity="0", condim="3", contype="1", friction=".4 .1 .1", solimp="0.0 0.8 0.01", solref="0.02 1", material="self")
ET.SubElement(default, "motor", ctrllimited="true", ctrlrange="-1 1")

# Add <size> element
ET.SubElement(root, "size", nstack="300000", nuser_geom="1")

# Add <option> element
ET.SubElement(root, "option", gravity="0 0 -9.81", timestep="0.01")

# Add <worldbody> element
worldbody = ET.SubElement(root, "worldbody")

# Add <light> element
ET.SubElement(worldbody, "light", cutoff="100", diffuse="1 1 1", dir="-1 -0 -1.5", directional="true", exponent="1", pos="0 0 1.3", specular=".1 .1 .1")

# Add ground geometry
ET.SubElement(worldbody, "geom", conaffinity="1", condim="3", material="grid", name="floor", pos="0 0 0", size="40 40 40", type="plane")

# Create torso body
torso = ET.SubElement(worldbody, "body", name="torso", pos="0 0 1.4")
ET.SubElement(torso, "geom", fromto=f"{-torso_half_length} 0 0 {torso_half_length} 0 0", name="torso", size="0.046", type="capsule")

# Add head geometry
ET.SubElement(torso, "geom", axisangle="0 1 0 .87", name="head", pos=f"{torso_half_length + 0.1} 0 .1", size="0.046 .15", type="capsule")

# Add root joints
ET.SubElement(torso, "joint", armature="0", axis="1 0 0", damping="0", limited="false", name="rootx", pos="0 0 0", stiffness="0", type="slide")
ET.SubElement(torso, "joint", armature="0", axis="0 0 1", damping="0", limited="false", name="rootz", pos="0 0 0", stiffness="0", type="slide")
ET.SubElement(torso, "joint", armature="0", axis="0 1 0", damping="0", limited="false", name="rooty", pos="0 0 0", stiffness="0", type="hinge")

# bthigh element
bthigh_pos_1 = bthigh_length * math.sin(-3.8)
bthigh_pos_2 = bthigh_length * math.cos(-3.8)
bthigh = ET.SubElement(torso, "body", name="bthigh", pos=f"{-torso_half_length} 0 0")
ET.SubElement(bthigh, "joint", axis="0 1 0", damping="6", name="bthigh", pos="0 0 0", range="-.52 1.05", stiffness="240", type="hinge")
ET.SubElement(bthigh, "geom", axisangle="0 1 0 -3.8", name="bthigh", pos=f"{bthigh_pos_1} 0 {bthigh_pos_2}", size=f"0.046 {bthigh_length}", type="capsule")

# bshin element
bshin_pos_1 = bshin_length * math.sin(-2.03)
bshin_pos_2 = bshin_length * math.cos(-2.03)
bthigh_2_bshin_pos_1 = 2 * bthigh_pos_1
bthigh_2_bshin_pos_2 = 2 * bthigh_pos_2
bshin = ET.SubElement(bthigh, "body", name="bshin", pos=f"{bthigh_2_bshin_pos_1} 0 {bthigh_2_bshin_pos_2}")
ET.SubElement(bshin, "joint", axis="0 1 0", damping="4.5", name="bshin", pos="0 0 0", range="-.785 .785", stiffness="180", type="hinge")
ET.SubElement(bshin, "geom", axisangle="0 1 0 -2.03", name="bshin", pos=f"{bshin_pos_1} 0 {bshin_pos_2}", size=f"0.046 {bshin_length}", type="capsule")

# Add cameras
ET.SubElement(torso, "camera", mode="trackcom", name="tilted", pos="3 3 0", xyaxes="1 -.9 0 0 0 -1")
ET.SubElement(torso, "camera", mode="trackcom", name="track", pos="0 5 0", xyaxes="1 0 0 0 0 -1")

# List joint & actuator.
joint_options = [("bthigh", 120), ("bshin", 90)]
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
    log.write(f"cheetah_3_back_{index}, range_len: {range_len}, disable_prob: {disable_prob}, "
              f"torso_length: {torso_length}, bthigh_length: {bthigh_length}, bshin_length: {bshin_length}, "
              f"bfoot_length: {bfoot_length}, fthigh_length: {fthigh_length}, fshin_length: {fshin_length}, "
              f"ffoot_length: {ffoot_length}, disabled_joints: {unselected_joints}\n")
    
# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define output file path
output_file = os.path.join(output_dir, f"cheetah_3_back_{index}.xml")

# Format and write XML file
xml_str = minidom.parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="  ")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(xml_str)

print(f"XML file generated and saved to {output_file}")
