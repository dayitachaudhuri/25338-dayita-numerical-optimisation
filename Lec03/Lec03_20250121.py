import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# 1. Setup the data
# ==============================

# Use Pandas to load data/real_estate_dataset.csv
df = pd.read_csv('data/real_estate_dataset.csv')

# Get the number of samples and features
n_samples, n_features = df.shape
print('Number of samples, features: {}, {}'.format(n_samples, n_features))

# Get the features
features = df.columns

# Save the column names to a text file 
with open('data/column_names.txt', 'w') as f:
    for feature in features:
        f.write(feature + '\n')

# ==============================================================================
# 2. Build a Linear Model to predict the price using some selected features in X
# ==============================================================================

print('\nBuilding a Linear Model to predict the price using some selected features in X')

# Use Square_Feet, Garage_Size, Location_Score, Distance_to_Center as features
X = df[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']].values

# Use Price as the target
y = df['Price'].values

# Get the number of samples and features in X
n_samples, n_features = X.shape

print('Shape of X: {}'.format(X.shape))
print('Datatype of X: {}'.format(X.dtype))

# Make and array of coefficients of the size of n_features+1. Initialize to 1.
coefs = np.ones(n_features + 1)

# Predict the price for each sample in X
predictions_bydefn = X @ coefs[1:] + coefs[0]

# Append a column of ones to X
X = np.hstack([np.ones((n_samples, 1)), X])

# Predict the price for each sample in X
predictions = X @ coefs

# See if all entries in predictions_bydefn and predictions are the same
is_same = np.allclose(predictions_bydefn, predictions)
print('Are the predictions the same? {}'.format(is_same))

# Calculate the error between the predictions and the actual prices
errors = y - predictions

# Calculate the relative error
relative_errors = errors / y

# Calculate the mean squared error using a loop
loss_loop = 0
for i in range(n_samples):
    loss_loop += (y[i] - predictions[i]) ** 2
mse = loss_loop / n_samples

# Calculate the mean squared error using matrix operations
loss_matrix = np.transpose(errors) @ errors / n_samples

# Check if the two ways of calculating the mean squared error are the same
is_same = np.allclose(mse, loss_matrix)
print('Are the mean squared errors the same? {}'.format(is_same))

# Print the size of the errors and its L2 norm
print('Size of errors: {}'.format(errors.shape))
print('L2 norm of errors: {}'.format(np.linalg.norm(errors)))

# Print the L2 norm of the relative errors
print('L2 norm of relative errors: {}'.format(np.linalg.norm(relative_errors)))

# Print the mean squared error
print('Mean Squared Error: {}'.format(mse))

# ========================================
# 3. What is the Optimisation Problem?
# ========================================
# We want to find the coefficients that minimise the mean squared error. 
# This is called the Least Squares Problem.

# Example - 
# Nu = \alpha * Re^m * Pr^n. 
# We want to find the values of \alpha, m, and n that minimise the error between the 
# predicted and actual values of Nu. To do so we take the log of both sides.
# log Nu = log \alpha + m * log Re + n * log Pr
# This is a linear equation in the coefficients. We can use the same approach as above to
# find the coefficients that minimise the error.

# What is a Solution?
# A solution is a set of coefficients that minimises the objective function.

# How do I find a Solution?
# By searching for the coefficients at which the gradient of the objective function is zero.
# Or we can set the gradient to zero and solve for the coefficients.

# ========================================
# 4. Solve the Least Squares Problem
# ========================================

print('\nSolution using the Normal Equations with Selected Features')

# Write the loss matrix in terms of the data and coefficients
loss_matrix = np.transpose(y - X @ coefs) @ (y - X @ coefs) / n_samples

# Calculate the gradient of the loss wrt the coefficients
grad_matrix = (-2/n_samples) * np.transpose(X) @ (y - X @ coefs) 

# We set grad_matrix to zero and solve for the coefficients
coefs = np.linalg.inv(X.T @ X) @ X.T @ y

# Save the coefficients to a csv file for viewing
np.savetxt('data/coeffs.csv', coefs, delimiter=',')

# Predict the price for each sample in X
predictions_model = X @ coefs

# Calculate the error between the predictions and the actual prices
errors_model = y - predictions_model

# Calculate the relative error
relative_errors_model = errors_model / y

# Print the L2 norm of the errors
print('L2 norm of errors: {}'.format(np.linalg.norm(errors_model)))
print('L2 norm of relative errors: {}'.format(np.linalg.norm(relative_errors_model)))

# ====================================================================================
# 5. Build a Linear Model to predict the price using all the features in the dataset
# ====================================================================================

print('\nSolution using Normal Equations with All Features')

# Use all the features in the dataset
X = df.drop('Price', axis=1).values
y = df['Price'].values

# Get the number of samples and features in X
n_samples, n_features = X.shape

# Append a column of ones to X
X = np.hstack([np.ones((n_samples, 1)), X])

# Calculate the coefficients that minimise the mean squared error
coefs = np.linalg.inv(X.T @ X) @ X.T @ y

# Save the coefficients to a csv file for viewing
np.savetxt('data/coeffs_all.csv', coefs, delimiter=',')

# Predict the price for each sample in X
predictions_model = X @ coefs

# Calculate the error between the predictions and the actual prices
errors_model = y - predictions_model

# Calculate the relative error
relative_errors_model = errors_model / y

# Print the L2 norm of the errors
print('L2 norm of errors: {}'.format(np.linalg.norm(errors_model)))
print('L2 norm of relative errors: {}'.format(np.linalg.norm(relative_errors_model)))

# ============================================================
# 6. Solve the Least Squares Problem using matrix decomposition
# ============================================================

# Calculate the rank of X.T @ X
rank = np.linalg.matrix_rank(X.T @ X)
print('Rank of X.T @ X: {}'.format(rank))

# ----------------------------------------------------------
# 6.1 Solve the Least Squares Problem using QR Factorisation
# ----------------------------------------------------------

print('\nSolution using QR Factorisation')

Q, R = np.linalg.qr(X)

print('Shape of Q: {}'.format(Q.shape))
print('Shape of R: {}'.format(R.shape))

# Write R to a csv file for viewing
np.savetxt('data/R.csv', R, delimiter=',')

# R @ coefs = b

# Check if Q is orthogonal
sol = Q.T @ Q
np.savetxt('data/sol.csv', sol, delimiter=',')

# X = QR
# X.T @ X = R.T @ Q.T @ Q @ R = R.T @ R
# X.T @ y = R.T @ Q.T @ y
# R @ coefs = Q.T @ y

b = Q.T @ y

print('Shape of b: {}'.format(b.shape))
print('Shape of R: {}'.format(R.shape))

# coefs_qr = np.linalg.inv(R) @ b

# Use a loop to solve R @ coefs = b using back substitution

coeffs_qr_loop = np.zeros(n_features + 1)

for i in range(n_features, -1, -1):
    coeffs_qr_loop[i] = b[i]
    for j in range(i + 1, n_features + 1):
        coeffs_qr_loop[i] -= R[i, j] * coeffs_qr_loop[j]
    coeffs_qr_loop[i] /= R[i, i]
    
# Save the coefficients to a csv file for viewing
np.savetxt('data/coeffs_qr_loop.csv', coeffs_qr_loop, delimiter=',')

# Predict the price for each sample in X
predictions_qr = X @ coeffs_qr_loop

# Calculate the error between the predictions and the actual prices
errors_qr = y - predictions_qr

# Calculate the L2 norm of the errors
print('L2 norm of errors (QR): {}'.format(np.linalg.norm(errors_qr)))

# Calculate the relative errors
relative_errors_qr = errors_qr / y

# Print the L2 norm of the relative errors
print('L2 norm of relative errors (QR): {}'.format(np.linalg.norm(relative_errors_qr)))

# ----------------------------------------------------------
# 6.2 Solve the Normal Equations using SVD Approach
# ----------------------------------------------------------

# Eigen Decomponsition of a Square Matrix
# A = V @ D @ V.T
# A^-1 = V @ D^-1 @ V.T
# X*coeffs = y
# A = X.T @ X

# Normal Equation: X.T @ X @ coeffs = X.T @ y
# Xdagger = (X.T @ X)^-1 @ X.T

print('\nSolution using SVD')

U, S, Vt = np.linalg.svd(X, full_matrices=False)

# (HOMEWORK) To Complete: Calculate the coeffs_svd using the pseudo inverse of X
pseudo_inv = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T

# Calculate the coefficients
coeffs_svd = pseudo_inv @ y

# Save the coefficients to a CSV file for viewing
np.savetxt('data/coeffs_svd_hw.csv', coeffs_svd, delimiter=',')

# Predict the price for each sample in X
predictions_svd = X @ coeffs_svd

# Calculate the error between the predictions and the actual prices
errors_svd = y - predictions_svd

# Calculate the L2 norm of the errors
print('L2 norm of errors (SVD): {}'.format(np.linalg.norm(errors_svd)))

# Calculate the relative errors
relative_errors_svd = errors_svd / y

# Print the L2 norm of the relative errors
print('L2 norm of relative errors (SVD): {}'.format(np.linalg.norm(relative_errors_svd)))

# (CLASS SOLUTION)

# Write X as product of U, S, and Vt
X_svd = U @ np.diag(S) @ Vt

# Normal Equation: X.T @ X @ coeffs = X.T @ y

# Solve for X_svd_T @ X_svd @ coeffs = X_svd_T @ y
# Replace X_svd with U @ np.diag(S) @ Vt
# Vt.T @ np.diag(S)^2 @ Vt @ coeffs = Vt.T @ np.diag(S) @ U.T @ y
# np.diag(S)^2 @ Vt @ coeffs = np.diag(S) @ U.T @ y
# coeffs = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y

coeffs_svd = Vt.T @ np.diag(1/S) @ U.T @ y
coeffs_svd_pinv = np.linalg.pinv(X) @ y

# Save the coefficients to a csv file for viewing
np.savetxt('data/coeffs_svd.csv', coeffs_svd, delimiter=',')
np.savetxt('data/coeffs_svd_pinv.csv', coeffs_svd_pinv, delimiter=',')

#================================================================
# 7. Plotting the Data and Regression Line
#================================================================

# Plot the data on X[:, 1] vs y
# Also plot the regression line with only X[:, 0] and X[:, 1] as features
# First make X[:1] as np.arange between min and max of X[:, 1]
# Then calculate the predictions using the coefficients from the SVD approach

X = df[['Square_Feet']].values
y = df['Price'].values
X = np.hstack([np.ones((n_samples, 1)), X])
 
coeff_1 = np.linalg.inv(X.T @ X) @ X.T @ y
predictions_1 = X @ coeff_1
X_feature = np.arange(np.min(X[:,1]), np.max(X[:,1]), 10)
print("\nMin of X[:,1]", np.min(X[:,1]))
print("Max of X[:,1]", np.max(X[:,1]))

# Pad the X_feature with ones
X_feature = np.hstack([np.ones((X_feature.shape[0], 1)), X_feature.reshape(-1, 1)])
plt.scatter(X[:,1], y, label='Data')
plt.plot(X_feature[:,1], X_feature  @ coeff_1, label='Regression Line', c='Red')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Price vs Square Feet')
plt.legend()
plt.show()
plt.savefig('images/Price_vs_Square_Feet.png')