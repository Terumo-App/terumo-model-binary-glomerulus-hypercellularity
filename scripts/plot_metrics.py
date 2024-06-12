import matplotlib.pyplot as plt
import numpy as np

lesions = [ 'accuracy',	'precision','recall','fscore']
# Podocytopathy
# CTEs = [0.85,	0.853535354,	0.85,	0.84962406]
# error = [0.0405,	0.0279,	0.050990195,	0.045232]

# #  Sclerosis
CTEs = [0.71,	0.713838384,	0.71, 0.708662566]
error = [0.02,	0.019699171,	0.02,	0.020400496]

x_pos = np.arange(len(lesions))

# Build the plot
fig, ax = plt.subplots()
# Plot the bar chart with error bars
bars = ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)

# Add labels to each bar
for bar, cte, err in zip(bars, CTEs, error):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height, f'{cte:.2f} ± {err:.2f}', ha='center', va='bottom')

# ax.set_ylabel('F1 Score ($\degree C^{-1}$)')
ax.set_xticks(x_pos)
ax.set_xticklabels(lesions)
ax.set_title('Sclerosis - Binary Classifiers Metrics')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('Sclerosis_metrics.png')
plt.show()