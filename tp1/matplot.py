import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 9, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a figure and an axis
plt.figure(figsize=(10, 6))

# Plot the data
plt.plot(x, y, label='y = 2x')

# Add title and labels
plt.title('Sample Plot')
plt.xlabel('X')
plt.ylabel('Y')

# Add a legend
plt.legend()

# Show the plot
plt.show()
