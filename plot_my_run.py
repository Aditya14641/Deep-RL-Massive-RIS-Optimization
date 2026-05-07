import os
import numpy as np
import matplotlib.pyplot as plt

folder = "./Learning Curves/custom/"
files = [f for f in os.listdir(folder) if f.endswith('.npy')]

if not files:
    print("No data found! Make sure the simulation finished running.")
    exit()

n_results = {}
p_results = {}

print("Scanning your M=256 data files...")
for file in files:
    parts = file.split('_')
    M, N, K, Pt = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
    
    # Load data and get the final converged reward (average of last 100 steps to smooth it out)
    data = np.load(os.path.join(folder, file)).squeeze()
    if data.ndim > 1: data = np.concatenate(data)
    final_reward = np.mean(data[-100:]) 
    
    # Sort into experiments
    if Pt == 20.0 and N in [10, 50, 100, 150, 200]: 
        n_results[N] = final_reward
    if N == 32 and Pt in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:    
        p_results[Pt] = final_reward

# --- Plot Figure 5 (Varying N) ---
if n_results:
    sorted_n = sorted(n_results.items())
    plt.figure(figsize=(8, 6))
    plt.plot([i[0] for i in sorted_n], [i[1] for i in sorted_n], color='green', marker='o', linewidth=2, label="Proposed DRL method (M=256)")
    plt.title("Figure 5 Recreated: Sum Rate vs Number of RIS Elements")
    plt.xlabel("Number of elements in RIS (N)")
    plt.ylabel("Sum rate (bps/Hz)")
    plt.grid(True)
    plt.legend()
    plt.savefig("My_Figure_5.png") # Saves a copy for your report!
    plt.show()

# --- Plot Figure 7 (Varying Pt) ---
if p_results:
    sorted_p = sorted(p_results.items())
    plt.figure(figsize=(8, 6))
    plt.plot([i[0] for i in sorted_p], [i[1] for i in sorted_p], color='blue', marker='s', linewidth=2, label="Proposed DRL method (M=256)")
    plt.title("Figure 7 Recreated: Sum Rate vs Transmit Power")
    plt.xlabel("Transmit Power Pt (dBm)")
    plt.ylabel("Sum rate (bps/Hz)")
    plt.grid(True)
    plt.legend()
    plt.savefig("My_Figure_7.png") # Saves a copy for your report!
    plt.show()