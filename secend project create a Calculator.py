import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
class Calculator:
    def __init__(self, numbers):
        self.numbers = numbers

    def add(self):
        return np.sum(self.numbers)

    def subtract(self):
        return np.subtract.reduce(self.numbers)

    def multiply(self):
        return np.prod(self.numbers)

    def divide(self):
        return np.divide.reduce(self.numbers)

    def mean(self):
        return np.mean(self.numbers)

    def median(self):
        return np.median(self.numbers)

    def variance(self):
        return np.var(self.numbers)

    def standard_deviation(self):
        return np.std(self.numbers)

    def normal_distribution(self):
        mean = self.mean()
        std_dev = self.standard_deviation()
        x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 100)
        y = scipy.stats.norm.pdf(x, mean, std_dev)
        plt.plot(x, y)
        plt.title("Normal Distribution")
        plt.xlabel("X")
        plt.ylabel("PDF")
        plt.show()

# Sample list of numbers
numbers = [2, 4, 6, 8, 10]

# Create a Calculator object
calculator = Calculator(numbers)

# Perform calculations
print("Sum:", calculator.add())
print("Difference:", calculator.subtract())
print("Product:", calculator.multiply())
print("Quotient:", calculator.divide())
print("Mean:", calculator.mean())
print("Median:", calculator.median())
print("Variance:", calculator.variance())
print("Standard Deviation:", calculator.standard_deviation())

# Plot the normal distribution of the numbers
calculator.normal_distribution()