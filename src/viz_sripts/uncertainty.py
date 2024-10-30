import matplotlib.pyplot as plt
import numpy as np

# Uncertainty measurement methods and models
methods = ['MC-Dropout', 'Deep Ens.', 'Bagged Ens.', 'Hyperparameter Ens.']
models = ['ProvDetector', 'SIGL', 'ThreaTrace', 'NodLink', 'MAGIC', 'Kairos', 'Flash', 'R-CAID']
num_bars_per_method = len(models)  # Number of bars for each method

# Sample data for demonstration (replace with actual data)
rho_ap_values = np.random.rand(len(methods), num_bars_per_method)  # Random data for ρ_AP
sigma_ap_values = np.random.rand(len(methods), num_bars_per_method)  # Random data for σ_AP

# Define Material Design colors for each model and color for sigma_AP
colors = ['#FF69B4', '#FF5722', '#4CAF50', '#3F51B5', '#FFC107', '#009688', '#673AB7', '#607D8B']  # Unique colors per model
metric2_color = 'gray'  # Gray color for σ_AP

# Define positions for the grouped bars within each method
group_width = 0.7  # Total width of each group of bars
bar_width = group_width / num_bars_per_method  # Width of individual bars within each method group
x_positions = {method: np.arange(num_bars_per_method) * bar_width - (group_width / 2) + j + 0.5 
               for j, method in enumerate(methods)}

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars for each method with smallest value at the bottom
for j, method in enumerate(methods):
    x = x_positions[method]
    for i in range(num_bars_per_method):
        # Get the values for rho_AP and sigma_AP
        rho_ap = rho_ap_values[j, i]
        sigma_ap = sigma_ap_values[j, i]
        
        # Plot the smaller value at the bottom and the larger on top
        if rho_ap < sigma_ap:
            ax.bar(x[i], rho_ap, width=bar_width, color=colors[i],  edgecolor='black', label=f'{models[i]} ρ_AP' if j == 0 else "")
            ax.bar(x[i], sigma_ap - rho_ap, width=bar_width, bottom=rho_ap, color=colors[i], alpha=0.5,  edgecolor='black', label=f'{models[i]} σ_AP' if j == 0 else "")
        else:
            ax.bar(x[i], sigma_ap, width=bar_width, color=colors[i], alpha=0.5,  edgecolor='black', label=f'{models[i]} σ_AP' if j == 0 else "")
            ax.bar(x[i], rho_ap - sigma_ap, width=bar_width, bottom=sigma_ap, color=colors[i],  edgecolor='black', label=f'{models[i]} ρ_AP' if j == 0 else "")

# Configure x-axis
ax.set_xticks([i + 0.5 for i in range(len(methods))])
ax.set_xticklabels(methods)
ax.set_xlabel('Uncertainty Measurement Methods')
ax.set_ylabel('AP Score')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Metrics by Model")

# Save and show plot
plt.tight_layout()
plt.savefig("b.png")
