import numpy as np

# Define the conditional probabilities
P_w = 0.20
P_x = 0.70
P_y_given_w = 0.15
P_y_given_not_w = 0.10
P_z_given_x_and_y = 0.65
P_z_given_x_and_not_y = 0.40
P_z_given_not_x_and_y = 0.60
P_z_given_not_x_and_not_y = 0.25

# Calculate the joint probability
P_w_and_x_and_not_y_and_z = P_w * P_x * (1 - P_y_given_w) * P_z_given_x_and_not_y

# Print the joint probability
print(P_w_and_x_and_not_y_and_z)