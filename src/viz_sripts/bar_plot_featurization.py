import matplotlib.pyplot as plt
import numpy as np

# Define methods and number of bars per method
methods = ['word2vec', 'word2vec+alacarte', 'fasttext', 'HFH', 'doc2vec']
num_bars_per_method = 6  # Number of bars within each method

# Sample data for Metric 1 and Metric 2 for each method
metric1_values = {method: np.random.rand(num_bars_per_method) * 0.6 for method in methods}
metric2_values = {method: np.random.rand(num_bars_per_method) * 0.4 for method in methods}

# Define Material Design colors for each of the 6 bars and the color for Metric 2
colors = ['#3F51B5', '#FF5722', '#FFC107', '#4CAF50', '#FF69B4', '#673AB7']  # Higher contrast Material colors

# Define positions for the grouped bars within each method
group_width = 0.7  # Total width of each group of 6 bars
bar_width = group_width / num_bars_per_method  # Width of individual bars within a group
x_positions = {method: np.arange(num_bars_per_method) * bar_width - (group_width / 2) + i + 0.5 
               for i, method in enumerate(methods)}

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for each method
for method, x in x_positions.items():
    for i in range(num_bars_per_method):
        # Get values for metric 1 and metric 2
        metric1 = metric1_values[method][i]
        metric2 = metric2_values[method][i]

        # Plot the two bars side by side
        ax.bar(x[i] - bar_width / 4, metric1, width=bar_width / 2, color=colors[i], edgecolor='black', 
               label="Train" if method == methods[0] and i == 0 else "")
        ax.bar(x[i] + bar_width / 4, metric2, width=bar_width / 2, color=colors[i], alpha=0.4, edgecolor='black', 
               label="Train+Test" if method == methods[0] and i == 0 else "")

# Configure the x-axis with method labels
ax.set_xticks([i + 0.5 for i in range(len(methods))])
ax.set_xticklabels(methods, fontsize=15)
ax.set_ylabel('AP Score', fontsize=15)
ax.legend(fontsize=15)

# Save and show plot
plt.tight_layout()
plt.savefig("a.png")
plt.show()