import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for the two normal distributions
mu1, sigma1 = -2, 0.05  # mean and standard deviation for first normal distribution
mu2, sigma2 = 2, 0.05 # mean and standard deviation for second normal distribution

# Create a range of x values
x = np.linspace(-4, 4, 1000)

# Compute the PDFs for both distributions
pdf1 = norm.pdf(x, mu1, sigma1)
pdf2 = norm.pdf(x, mu2, sigma2)

# Weights for the two distributions (these should sum to 1)
weight1 = 0.5
weight2 = 0.5

# Create the bimodal distribution by combining the two PDFs
bimodal_pdf = weight1 * pdf1 + weight2 * pdf2

# Plotting
plt.plot(x, bimodal_pdf, label='P(State | F/T Reading)')
plt.fill_between(x, bimodal_pdf, alpha=0.5)
plt.axvline(x=mu1, color='r', linestyle='--', label='Contact with Left Wall')
plt.axvline(x=mu2, color='g', linestyle='--', label='Contact with Right Wall')

# Labels and Title
plt.title('Bimodal Probability Density Plot')
plt.xlabel('Tip Location')
plt.ylabel('Probabilty')
plt.legend(loc = 'upper right')

# Show the plot
plt.show()
