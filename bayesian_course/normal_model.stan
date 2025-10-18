
data {
  int<lower=0> N;       // Number of data points
  array[N] real y;       // Observations (new syntax)
}
parameters {
  real mu;              // Mean parameter
  real<lower=0> sigma;  // Standard deviation parameter
}
model {
  y ~ normal(mu, sigma);  // Likelihood
}
