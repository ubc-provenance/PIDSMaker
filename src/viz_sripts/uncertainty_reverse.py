import matplotlib.pyplot as plt
import numpy as np

# Define models (systems) and methods
models = ['ProvDetector', 'SIGL', 'ThreaTrace', 'NodLink', 'MAGIC', 'Kairos', 'Flash', 'R-CAID']
methods = ['MC-Dropout', 'Deep Ens.', 'Bagged Ens.', 'Hyperparameter Ens.']

# Sample data for demonstration (replace with actual data)
rho_ap_values = np.random.rand(len(models), len(methods))  # Random data for ρ_AP (models x methods)
sigma_ap_values = np.random.rand(len(models), len(methods))  # Random data for σ_AP (models x methods)

# Define color groups for each method
method_colors = {
    'MC-Dropout': '#F48FB1',       # Pink 300
    'Deep Ens.': '#F06292',        # Pink 400
    'Bagged Ens.': '#FF8A65',      # Deep Orange 300
    'Hyperparameter Ens.': '#FF7043' # Deep Orange 400
}
colors = ["#4374aa", "#c37e4b"]

# Define plot parameters
group_width = 0.7  # Total width of the group of bars for each model
bar_width = group_width / len(methods)  # Width of each bar within the group

# Define positions for each system (model)
x = np.arange(len(models))

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Loop through each method and model to plot bars
for j, method in enumerate(methods):
    for i, model in enumerate(models):
        # Get the values for ρ_AP and σ_AP
        rho_ap = rho_ap_values[i, j]
        sigma_ap = sigma_ap_values[i, j]
        
        # Position each bar within its group
        x_pos = x[i] - group_width / 2 + j * bar_width + bar_width / 2
        
        alpha = (len(methods) - j/2) / len(methods)
        # Plot the bars with gradient effect using the same color with different alpha
        if rho_ap < sigma_ap:
            ax.bar(x_pos, rho_ap, width=bar_width, color=colors[0], alpha=alpha, edgecolor='black',label=f'ρ_AP {method}' if i == 0 else "")
            ax.bar(x_pos, sigma_ap - rho_ap, width=bar_width, bottom=rho_ap, color=colors[1], alpha=alpha, edgecolor='black',label=f'σ_AP {method}' if i == 0 else "")
        else:
            ax.bar(x_pos, sigma_ap, width=bar_width, color=colors[1], alpha=alpha, edgecolor='black',label=f'σ_AP {method}' if i == 0 else "")
            ax.bar(x_pos, rho_ap - sigma_ap, width=bar_width, bottom=sigma_ap, color=colors[0], alpha=alpha, edgecolor='black',label=f'ρ_AP {method}' if i == 0 else "")

# Configure x-axis with model names
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
# ax.set_xlabel('Models (Systems)')
ax.set_ylabel('Uncertainty Metric (%AP)')
# ax.set_title('Uncertainty Measurements by Model and Method')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Metrics by Method")

# Save and show plot
plt.tight_layout()
plt.savefig("b.png")
plt.show()