import matplotlib.pyplot as plt
import numpy as np

lesions = [ 'accuracy',	'precision','recall','fscore']
# Podocytopathy
# CTEs = [0.72,	0.765554299,	0.72,	0.705566924]
# error = [0.050990195, 0.037768001,	0.050990195, 0.065332328]

#  Sclerosis
CTEs = [0.6125,	0.619724026, 0.6125, 0.602148126]
error = [0.07395099728874518, 0.076152126, 0.073950997,	0.081664231]

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