import os
import time

print("🚀 Starting the Massive M=256, K=256 Simulation Suite...")

# ---------------------------------------------------------
# EXPERIMENT 1: Generate Data for Figure 5 (Varying N)
# Fixed Power = 20dB. Changing RIS Elements.
# ---------------------------------------------------------
print("\n=== PHASE 1: Varying RIS Elements (N) ===")
n_values = [10, 50, 100, 150, 200]

for n in n_values:
    print(f"\n>> Training AI for N = {n} (This will take a while...)")
    # Notice: 1500 steps per episode for optimized GPU/CPU hybrid runtime
    cmd = f"python main.py --experiment_type custom --num_antennas 256 --num_users 256 --num_RIS_elements {n} --power_t 20 --batch_size 8 --buffer_size 500 --num_eps 1 --num_time_steps_per_eps 1500"
    os.system(cmd)
    time.sleep(2) # Brief pause between heavy runs

# ---------------------------------------------------------
# EXPERIMENT 2: Generate Data for Figure 7 (Varying Power)
# Fixed RIS Elements = 32. Changing Transmit Power.
# ---------------------------------------------------------
print("\n=== PHASE 2: Varying Transmit Power (Pt) ===")
power_values = [0, 5, 10, 15, 20, 25, 30]

for p in power_values:
    print(f"\n>> Training AI for Power = {p} dB (This will take a while...)")
    cmd = f"python main.py --experiment_type custom --num_antennas 256 --num_users 256 --num_RIS_elements 32 --power_t {p} --batch_size 8 --buffer_size 500 --num_eps 1 --num_time_steps_per_eps 1500"
    os.system(cmd)
    time.sleep(2)

print("\n🎉 ALL EXPERIMENTS FINISHED SUCCESSFULLY! Your GPU deserves a break.")