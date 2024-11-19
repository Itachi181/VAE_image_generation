import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = "epoch_losses_2.csv"
data = pd.read_csv(csv_file)

# Extract columns
epochs = data['epoch']
total_loss = data['total_loss']

# Plot the data with a logarithmic scale
plt.figure(figsize=(10, 6))
plt.plot(epochs, total_loss, marker='o', label='Total Loss', linestyle='-', linewidth=2)
plt.yscale('log')  # Apply logarithmic scale to the y-axis
plt.xlabel('Epoch')
plt.ylabel('Total Loss (log scale)')
plt.title('Epoch vs. Total Loss (Log Scale)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save the plot as an image
plot_filename = "epoch_vs_total_loss_log_scale.png"
plt.savefig(plot_filename, dpi=300)

# Show the plot
plt.show()

print(f"Plot saved as {plot_filename}")


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV file
# csv_file = "epoch_losses_2.csv"
# data = pd.read_csv(csv_file)

# # Filter data to include only rows where the epoch is a multiple of 500
# interval = 350
# filtered_data = data[data['epoch'] % interval == 0]

# # Extract columns
# epochs = filtered_data['epoch']
# total_loss = filtered_data['total_loss']

# # Plot the data
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, total_loss, marker='o', label='Total Loss', linestyle='-', linewidth=2)
# plt.xlabel('Epoch')
# plt.ylabel('Total Loss')
# plt.title(f'Epoch vs. Total Loss (Every {interval} Epochs)')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()
# plt.tight_layout()

# # Save the plot as an image
# plot_filename = f"epoch_vs_total_loss_{interval}_interval.png"
# plt.savefig(plot_filename, dpi=300)

# # Show the plot
# plt.show()

# print(f"Plot saved as {plot_filename}")
