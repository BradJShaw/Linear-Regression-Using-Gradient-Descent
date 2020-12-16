# Linear-Regression-Using-Gradient-Descent
A Linear-Regression Model built from scratch using the gradient descent Algorithm with RMSprop optimization.

# Objective
Train the Linear Regression Model with a Student Performance Dataset (https://archive.ics.uci.edu/ml/datasets/Student+Performance) in order to predict the final year grade for a student's math course.

# Math
Error Function (MSE):

![](https://github.com/BradJShaw/Linear-Regression-Using-Gradient-Descent/blob/main/equations/Error%20Function.png)

Above is the formulas for calculating the MSE.

We can represent this algorithm using linear algebra in order to calculate the error of all the predictions without looping.

![](https://github.com/BradJShaw/Linear-Regression-Using-Gradient-Descent/blob/main/equations/j(theta).png)
![](https://github.com/BradJShaw/Linear-Regression-Using-Gradient-Descent/blob/main/equations/h(x)_Matrix.png)

________________________
Weight Update (Gradient Descent with RMSprop):

![](https://github.com/BradJShaw/Linear-Regression-Using-Gradient-Descent/blob/main/equations/RMSprop.png)

Above is the formula for updating the weights.

Like the Error Function earlier, we can represent this algorithm using linear algebra in order to update the weights without looping.

![](https://github.com/BradJShaw/Linear-Regression-Using-Gradient-Descent/blob/main/equations/RMSpropLinearAlgebra.png)

___________________________________
Learning Rate (RMSprop):

![](https://github.com/BradJShaw/Linear-Regression-Using-Gradient-Descent/blob/main/equations/learningRate.png)

Above is the formula for update the learning rate as the ai trains.

Again, We can rewrite this using Linear Algebra.

![](https://github.com/BradJShaw/Linear-Regression-Using-Gradient-Descent/blob/main/equations/learningRateLinearAlgebra.png)

# Graphs
Heat map of the dataset: 

![](https://github.com/BradJShaw/Linear-Regression-Using-Gradient-Descent/blob/main/graphs/heatmap.png)

_____________
Students vs Grades (before outliers taken out):

![](https://github.com/BradJShaw/Linear-Regression-Using-Gradient-Descent/blob/main/graphs/Student%20vs%20Grades%20(with%20outliers).png)

___________
Students vs Grade (after outliers taken out):

![](https://github.com/BradJShaw/Linear-Regression-Using-Gradient-Descent/blob/main/graphs/Student%20vs%20Grades%20(without%20outliers).png)

_____________
Correlations with Final Grade in ascending order:

![](https://github.com/BradJShaw/Linear-Regression-Using-Gradient-Descent/blob/main/graphs/correlations.png)

# Results

R2: 0.933

MAE: 0.607

MSE: 0.607

RMSE: .779
________
Error vs Iterations

![](https://github.com/BradJShaw/Linear-Regression-Using-Gradient-Descent/blob/main/equations/error%20vs%20iterations.png)

__________
Percentage vs Error

![](https://github.com/BradJShaw/Linear-Regression-Using-Gradient-Descent/blob/main/equations/predictions%20over%20actual.png)
