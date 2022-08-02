import matplotlib.pyplot as plt
import numpy as np

samples = np.random.normal(0, 1., 100000)


true_mean = np.mean(samples)
true_second_moment = np.sum((samples - true_mean) ** 2)
true_third_moment = np.sum((samples - true_mean) ** 3)
true_third_moment_abs = np.sum(np.abs(samples - true_mean) ** 3)
true_fourth_moment = np.sum((samples - true_mean) ** 4)


mean = 0.
second_moment = 0.
third_moment = 0.
third_moment_abs = 0.
fourth_moment = 0.

for n, sample in enumerate(samples):
    
    delta = sample - mean
    mean = (n * mean + sample) / (n + 1)


    fourth_moment = fourth_moment + delta ** 4 * (n ** 3 - n ** 2 + n) / (n + 1) ** 3 + 6 * delta ** 2 * second_moment / (n + 1) ** 2 - 4 * delta * third_moment / (n + 1)
    third_moment = third_moment + delta ** 3 * (n ** 2 - n) / (n + 1) ** 2 - 3 * second_moment * delta / (n + 1)
    
    third_moment_abs = third_moment_abs + abs(delta ** 3 * (n ** 2 - n) / (n + 1) ** 2 - 3 * second_moment * delta / (n + 1))
    second_moment = second_moment + delta ** 2 * (n) / (n + 1)


print(f'Mean: {true_mean}, {mean}')
print(f'Second Moment: {true_second_moment}, {second_moment}')
print(f'Third Moment: {true_third_moment}, {third_moment}')
print(f'Third Moment Abs: {true_third_moment_abs}, {third_moment_abs}')
print(f'Fourth Moment: {true_fourth_moment}, {fourth_moment}')
