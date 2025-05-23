import numpy as np
from matplotlib import pyplot as plt
import os
import argparse

# matplot setting.
def SetPlotRC():
    '''
    For ICML, NIPS etc. camera ready requirement
    '''
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)

SetPlotRC()
colors = ['seagreen', 'cornflowerblue', 'orange', 'gray', 'plum', 'lightcoral', 'dodgerblue']

def parse_args():
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument(
        "--xml_ids", 
        type=str, 
        nargs='+', 
        required=True, 
        help="List of XML identifiers (e.g., 'Cheetah (10 kinds)', 'Cheetah (100 kinds)')"
    )
    parser.add_argument(
        "--num_exp", 
        type=int, 
        required=True, 
        help="Number of experiments per XML identifier"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Directory containing experiment data"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    xml_ids = args.xml_ids
    num_exp = args.num_exp
    data_dir = args.data_dir

    # 动态生成 id_exps
    id_exps = [range(i * num_exp + 1, (i + 1) * num_exp + 1) for i in range(len(xml_ids))]

    print("xml_ids:", xml_ids)
    print("id_exps:", id_exps)
    print("data_dir:", data_dir)

    fig, ax = plt.subplots()
    for ind, xml_id in enumerate(xml_ids):
        exps = id_exps[ind]
        curve_data = []
        for exp in exps:
            run_data = np.load(os.path.join(data_dir, f'EXP_{exp:04d}.npy'))
            curve_data.append(run_data)
        min_len = min([run_data.shape[-1] for run_data in curve_data])
        curve_data = np.array([run_data[:, :min_len] for run_data in curve_data])  # [runs #, (steps, rewards), min_len]
        steps = curve_data[:, 0, :]  # [runs #, min_len]
        rewards = curve_data[:, 1, :]  # [runs #, min_len]
        # take average and std across all runs
        steps_mean = np.mean(steps, axis=0).astype(int)
        rewards_mean = np.mean(rewards, axis=0)
        rewards_std = rewards.std(axis=0) / np.sqrt(rewards.shape[0])
        # set up x-axis with obtained timesteps
        x_axis = np.array([(steps_mean.max() // rewards_mean.size) * i for i in range(rewards_mean.size)])
        # plot
        ax.plot(x_axis, rewards_mean, color=colors[ind % len(colors)], label=xml_ids[ind])
        plt.fill_between(x_axis, rewards_mean + rewards_std, rewards_mean - rewards_std, color=colors[ind % len(colors)], alpha=0.2)

    # global setting for each subplot
    plt.title('cheetah++', fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    plt.grid(True, which='both')
    plt.xlabel('Training Steps', fontsize=15)
    plt.ylabel('Mean Rewards', fontsize=15)
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', ncol=5, handlelength=1.8, fontsize=15, borderaxespad=1, borderpad=0.5, columnspacing=0.5, bbox_to_anchor=[0.6, 0.14])
    fig.subplots_adjust(bottom=0.25)
    output_path = os.path.join(data_dir, "output.png")
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    main()