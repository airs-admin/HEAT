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
parser.add_argument("--log_file", type=str, default="walker_xml.log", help="File to log generated parameters.")
args = parser.parse_args()
disable_prob = args.disable_prob
range_len = args.range_len
index = args.index
output_dir = args.output_dir
log_file = args.log_file

# Randomly generate dimensions for different body parts
torso_base = 0.6
left1_base = 0.48
left2_base = 0.384
left3_base = 0.2
right1_base = 0.48
right2_base = 0.384
right3_base = 0.2
torso_length = max(0.001, round(torso_base + (random.gauss(0, 1) * torso_base * range_len), 3))
torso_half_length = torso_length / 2
left1_length = max(0.001, round(left1_base + (random.gauss(0, 1) * left1_base * range_len), 3))
left2_length = max(0.001, round(left2_base + (random.gauss(0, 1) * left2_base * range_len), 3))
left3_length = max(0.001, round(left3_base + (random.gauss(0, 1) * left3_base * range_len), 3))
right1_length = max(0.001, round(right1_base + (random.gauss(0, 1) * right1_base * range_len), 3))
right2_length = max(0.001, round(right2_base + (random.gauss(0, 1) * right2_base * range_len), 3))
right3_length = max(0.001, round(right3_base + (random.gauss(0, 1) * right3_base * range_len), 3))
left1_2_torso_height = torso_length + left1_length + 0.136
right1_2_torso_height = torso_length + 0.136
torso_height = max(left1_2_torso_height, right1_2_torso_height)

# Create root element <mujoco>
root = ET.Element("mujoco", model="walker_generic")

# Include necessary files
ET.SubElement(root, "include", file="../../misc/skybox.xml")
ET.SubElement(root, "include", file="../../misc/visual.xml")
ET.SubElement(root, "include", file="../../misc/materials.xml")

# Add <compiler> element
ET.SubElement(root, "compiler", angle="degree", coordinate="global", inertiafromgeom="true")

# Add <default> element
default = ET.SubElement(root, "default")
ET.SubElement(default, "joint", armature="0.01", damping=".1", limited="true")
ET.SubElement(default, "geom", conaffinity="0", condim="3", contype="1", density="1000", friction=".7 .1 .1", material="self")

# Add <option> element
ET.SubElement(root, "option", integrator="RK4", timestep="0.002")

# Add <worldbody> element
worldbody = ET.SubElement(root, "worldbody")
ET.SubElement(worldbody, "light", cutoff="100", diffuse="1 1 1", dir="-1 -0 -1.5", directional="true", exponent="1", pos="0 0 1.3", specular=".1 .1 .1")
ET.SubElement(worldbody, "geom", conaffinity="1", condim="3", material="grid", name="floor", pos="0 0 0", size="40 40 40", type="plane")

# Create torso element
torso = ET.SubElement(worldbody, "body", name="torso", pos=f"0 0 {torso_height}")
ET.SubElement(torso, "camera", mode="trackcom", name="tilted", pos="2.7 3 1", xyaxes="1 -.9 0 0 0 -1")
ET.SubElement(torso, "camera", mode="trackcom", name="track", pos="0 5 1.5", xyaxes="1 0 0 0 0 -1")
ET.SubElement(torso, "joint", armature="0", axis="1 0 0", damping="0", limited="false", name="rootx", pos="0 0 0", stiffness="0", type="slide")
ET.SubElement(torso, "joint", armature="0", axis="0 0 1", damping="0", limited="false", name="rootz", pos="0 0 0", ref=f"{torso_height - torso_half_length}", stiffness="0", type="slide")
ET.SubElement(torso, "joint", armature="0", axis="0 1 0", damping="0", limited="false", name="rooty", pos=f"0 0 {torso_height - torso_half_length}", stiffness="0", type="hinge")
ET.SubElement(torso, "geom", friction="0.9", fromto=f"0 0 {torso_height} 0 0 {torso_height - torso_length}", name="torso_geom", size="0.07", type="capsule")

# Left leg structure
left1 = ET.SubElement(torso, "body", name="left1", pos=f"0 0 {torso_height - torso_length}")
ET.SubElement(left1, "geom", friction="0.9", fromto=f"0 0 {torso_height - torso_length} 0 0 {torso_height - torso_length - left1_length}", name="left1_geom", size="0.056", type="capsule")
ET.SubElement(left1, "joint", axis="0 -1 0", name="left1_joint", pos=f"0 0 {torso_height - torso_length}", range="-75 75", type="hinge")

# Add actuator.
actuator = ET.SubElement(root, "actuator")
ET.SubElement(actuator, "motor", ctrllimited="true", ctrlrange="-1.0 1.0", gear="100", joint="left1_joint")

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define output file path
output_file = os.path.join(output_dir, f"walker_2_main_{index}.xml")

# Format XML content
xml_str = minidom.parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="  ")

# Write XML content to file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(xml_str)

print(f"XML file generated and saved to {output_file}")
