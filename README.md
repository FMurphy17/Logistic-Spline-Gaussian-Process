# Logistic-Spline-Gaussian-Process

The Logistic Spline Gaussian Process (LSGP) model for in-vitro dissolution testing uses GPs to estimate dissolution curves given 
dissolution percentage data. This model incorporates a montonically-increasing prior by embedding a logistic function into the mean and Integrated q-fold Wiener kernel (q=2). This model additionally supports the inclusion of dissolution covariates (e.g. medium, medium viscosity, apparatus velocity) to enhance predictions.