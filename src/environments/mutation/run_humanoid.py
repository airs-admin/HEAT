import argparse
import os
import subprocess

def run_humanoid_scripts(num_runs, disable_prob, range_len, output_dir, log_file):
    # Define the script directory
    script_dir = "."

    # Get all humanoid script files
    humanoid_scripts = [
        f for f in os.listdir(script_dir)
        if f.startswith("random_humanoid_") and f.endswith(".py")
    ]

    if not humanoid_scripts:
        print("No humanoid scripts found in the specified directory.")
        return

    # Sort scripts by name to ensure consistent order
    humanoid_scripts.sort()

    for run in range(num_runs):
        for script in humanoid_scripts:
            script_path = os.path.join(script_dir, script)
            cmd = ["python", script_path, 
                   "--index", str(run + 1), 
                   "--disable_prob", str(disable_prob), 
                   "--range_len", str(range_len),
                   "--output_dir", str(output_dir),
                   "--log_file", str(log_file)]
            print(f"Running {script} with index {run + 1}: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple random humanoid generation scripts.")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs per script.")
    parser.add_argument("--disable_prob", type=float, default=0.0, help="Probability of disabling a joint actuator.")
    parser.add_argument("--range_len", type=float, default=0.1, help="Range length for body part length variation.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save generated XML files.")
    parser.add_argument("--log_file", type=str, default="humanoid_xml.log", help="File to log generated parameters.")
    args = parser.parse_args()

    run_humanoid_scripts(args.num_runs, args.disable_prob, args.range_len, args.output_dir, args.log_file)
