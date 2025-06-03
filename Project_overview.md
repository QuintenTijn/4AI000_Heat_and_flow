# Assignment:

**Problem statemtent:**
What machine learning model has the best performance in terms of speed and accuracy in predicting the reflection and accomodation coefficients of Hydrogen gas particles off of a interface?


## Method
We make different machine learning models to predict the output translational_velocity and angular_velocity after the gas particles off of the wall.
We test the performance of the models on the data we have and report on the results and which model ends up being the best for the task.

we will iteratively build each model by having it perform the following model tasks:
    A. The first iteration of the model will predict the reflection of the translational velocities without any interaction between particles.
    B. The second iteration of the model will predict the reflection with the particle rotational velocities taken into account as well.
    C. The third iteration will predict the accommodation coefficients.

After each model has been built it will be compared against the other ML models and mathematical models.


**Models and data to plot against eachother**

- MD data
- CLL mathematical model
- GMM ML model
- Bayesian ML model
- CNN ML model

**Plots for the final report**
- Scatter heatmap
- PDF line plot
- translational accomodation coefficient prediction - ground truth vs nr_components line plot
- Angular velocity input vs angular velocity output
- Calculate the angular momentum accomodation coefficients
- Model performance: training time, nr_components


