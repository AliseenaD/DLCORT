import numpy as np
import matplotlib.pyplot as plt

# Data
groups = ['1000-1,1000-2,524B_LR', '507x-1,2,3', '510F-1,2,3']
values = [
    [-0.177143, 0.287234, -0.859813],  # 1000/524B group (1000_1_d2, 1000_2_d2, 524B_LR_d2)
    [-0.16, -0.353718, -0.262626],      # 507X group (507X_1_d2, 507X_2_d2, 507X_3_d2)
    [-0.246154, 0.155174, -0.538462]   # 510F group (510F_1_d2, 510F_2_d2, 510F_3_d2)
]

# Calculate means and standard errors
means = [np.mean(row) for row in values]
sems = [np.std(row, ddof=1) / np.sqrt(len(row)) for row in values]

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
x = np.arange(len(groups))
bars = ax.bar(x, means, yerr=sems, capsize=5, alpha=0.7, color='skyblue', ecolor='black')

# Plot individual points
for i, row in enumerate(values):
    ax.scatter(np.full_like(row, i), row, color='navy', alpha=0.6, zorder=3)

# Customize plot
ax.set_ylabel('Discrimination Index')
ax.set_title('Average Discrimination Indices by Group with Error Bars D2 2Min')
ax.set_xticks(x)
ax.set_xticklabels(groups, rotation=45, ha='right')
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.set_ylim(bottom=-0.9, top=0.4)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show plot
plt.show()