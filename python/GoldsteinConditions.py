import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# Keep the existing constants but adjust them for better visualization
fx = 2
gradfpx = -2
c1 = 0.25
c2 = 0.75

# Use the existing alpha values
alpha = np.linspace(0, 1.5, 100)

# Calculate function values with a more intuitive quadratic function
f_alpha = fx + gradfpx * alpha + 1.0 * alpha**2
upper_bound = fx + c1 * gradfpx * alpha
lower_bound = fx + c2 * gradfpx * alpha

# Set the style
plt.style.use(['science', 'ieee'])

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot main function
ax.plot(alpha, f_alpha, label='$f(\\alpha) = f(x) + \\nabla f(x)^T p \\alpha + \\frac{1}{2} \\alpha^2$', color='blue')

# Plot upper and lower bounds
ax.plot(alpha, upper_bound, '--', label='Upper Bound', color='red', linewidth=1)
ax.plot(alpha, lower_bound, '--', label='Lower Bound', color='red', linewidth=1)
ax.fill_between(alpha, lower_bound, upper_bound, color='red', alpha=0.1, label='Sufficient Decrease')
ax.fill_between(alpha, f_alpha, upper_bound, where=(f_alpha < upper_bound), color='green', alpha=0.1, label='$\\nabla f(\\alpha) \\geq c_1 \\nabla f(0)$')
ax.fill_between(alpha, lower_bound, f_alpha, where=(f_alpha < lower_bound), color='blue', alpha=0.1, label='$\\nabla f(\\alpha) \\leq c_2 \\nabla f(0)$')


# Add h and v lines
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# Add arrows and annotations
ax.annotate('Increase $\\alpha$', xy=(1, 1.5), xytext=(1.2, 1.8), arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate('Decrease $\\alpha$', xy=(1, 0.5), xytext=(1.2, 0.8), arrowprops=dict(facecolor='black', arrowstyle='->'))

# Set labels and title
ax.set_xlabel('Step Size $\\alpha$')
ax.set_title('Goldstein Conditions')


# Add legend
ax.legend( loc='upper right', fontsize=8, frameon=True, framealpha=1, edgecolor='black', fancybox=False)
ax.grid(True)
ax.set_axisbelow(True)

# Save the figure
# plt.savefig('figures/goldstein_conditions.png', dpi=300, bbox_inches='tight')

plt.show()

