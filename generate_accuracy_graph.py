import matplotlib.pyplot as plt
import numpy as np

# Define models and their corresponding hypothetical accuracies
models = ['Facial\n(XCeption)', 'Speech\n(LSTM)', 'Text\n(BERT)', 'Multimodal\nFusion\n(Majority Voting)']
accuracies = [66.0, 99.2, 68.8, 99.6] # Percentages

# Define colors for the bars
colors = ['#64B5F6', '#4DB6AC', '#FFB74D', '#E57373']

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create the bar chart
bars = ax.bar(models, accuracies, color=colors, width=0.6, edgecolor='black', linewidth=1.2)

# Add data labels on top of the bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=12, fontweight='bold')

# Customize the chart
ax.set_ylim(0, 100) # Set y-axis limit to 100%
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('Performance Comparison: Single Modalities vs Multimodal Fusion', fontsize=16, fontweight='bold', pad=20)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=11)

# Add a subtle grid
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Highlight the Fusion bar by adding a distinct hatch or outline if desired
bars[-1].set_linewidth(2.5)
bars[-1].set_edgecolor('#B71C1C')

# Adjust layout and save the image
plt.tight_layout()
plt.savefig('accuracy_improvement.png', dpi=300, bbox_inches='tight')
print("Accuracy graph saved to accuracy_improvement.png")
