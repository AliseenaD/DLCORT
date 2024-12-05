import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Data organization
day1_left = [47.93, 39.33, 49.47]  # 510F_1, 510F_2, 510F_3
day1_right = [42.27, 44.13, 47.13]
day2_left = [28.73, 19.53, 42.07]
day2_right = [42.6, 28.4, 43.67]

# Calculate means
means = [np.mean(day1_left), np.mean(day1_right), np.mean(day2_left), np.mean(day2_right)]

# Calculate standard errors
sems = [stats.sem(day1_left), stats.sem(day1_right), stats.sem(day2_left), stats.sem(day2_right)]

# Set up the plot
plt.figure(figsize=(12, 6))
bar_width = 0.35
x = np.array([1, 2, 4, 5])  # Position bars with a gap between days

# Create bars with error bars
bars = plt.bar(x, means, bar_width, 
               color=['blue', 'red', 'blue', 'red'],
               alpha=0.7,
               yerr=sems,
               capsize=5)

# Add individual points
mouse_colors = ['red', 'blue', 'green']
mouse_labels = ['510F_1', '510F_2', '510F_3']

# Plot points for each bar
for i, data in enumerate([day1_left, day1_right, day2_left, day2_right]):
    for j, value in enumerate(data):
        jitter = np.random.normal(0, 0.02)
        if i == 0:  # Only add labels for the first set of points
            plt.scatter(x[i] + jitter, value, color=mouse_colors[j], s=50, alpha=0.8, 
                       label=mouse_labels[j])
        else:
            plt.scatter(x[i] + jitter, value, color=mouse_colors[j], s=50, alpha=0.8)

# Customize the plot
plt.ylabel('Time (seconds)')
plt.title('Mouse Performance - Left vs Right Object')
plt.xticks(x, ['Day 1\nLeft', 'Day 1\nRight', 'Day 2\nLeft', 'Day 2\nRight'])
plt.legend(title='Mouse ID')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust y-axis to start from 0
plt.ylim(bottom=0)

# Show the plot
plt.tight_layout()
plt.show()

# Print the means and SEMs for reference
for i, (mean, sem) in enumerate(zip(means, sems)):
    condition = ['Day 1 Left', 'Day 1 Right', 'Day 2 Left', 'Day 2 Right'][i]
    print(f"{condition} Mean ± SEM: {mean:.2f} ± {sem:.2f} seconds")