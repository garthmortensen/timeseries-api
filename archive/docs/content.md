# Market Risk Analysis

| Section    | Title                                      | Description                                                                                  |
|------------|--------------------------------------------|----------------------------------------------------------------------------------------------|
| I.1        | Basic Calculus for Finance                | Using basic math tools (like derivatives/integrals) to tackle finance problems.             |
| I.1.1      | Introduction                              | Quick overview of how calculus concepts help analyze financial data.                        |
| I.1.2      | Functions and Graphs, Equations and Roots | How to plot equations and find where they cross zero.                                        |
| I.1.2.1    | Linear and Quadratic Functions            | Straight lines vs. parabolas, used for simple financial formulas.                           |
| I.1.2.2    | Continuous and Differentiable Functions   | Smooth functions you can take derivatives of.                                               |
| I.1.2.3    | Inverse Functions                         | Flipping inputs and outputs, like reversing percentage returns.                             |
| I.1.2.4    | The Exponential Function                  | Fast-growing function often seen in compound interest.                                      |
| I.1.2.5    | The Natural Logarithm                     | The “undo” function for exponentials, used for log returns.                                 |
| I.1.3      | Differentiation and Integration           | How to measure slopes and areas, crucial for rates of change and totals.                   |
| I.1.3.1    | Definitions                               | The formal meaning of derivatives (instantaneous rate) and integrals (accumulated sum).     |
| I.1.3.2    | Rules for Differentiation                 | Shortcuts to find derivatives quickly (product rule, chain rule, etc.).                    |
| I.1.3.3    | Monotonic, Concave, and Convex Functions  | Shapes of functions—rising or falling, bowl-shaped or hill-shaped.                         |
| I.1.3.4    | Stationary Points and Optimization        | Spots where slope is zero, used to find maxima/minima (like profit peaks).                  |
| I.1.3.5    | Integration                               | Adding up tiny slices, useful for total change or area under curves (like cumulative returns). |
| I.1.4      | Analysis of Financial Returns             | Using these calculus ideas for gains, losses, and time-based returns.                      |
| I.1.4.1    | Discrete and Continuous Time Notation     | Difference between counting in steps vs. flowing in real time.                             |
| I.1.4.2    | Portfolio Holdings and Portfolio Weights | How much you hold of each asset, expressed as percentages.                                  |
| I.1.4.3    | Profit and Loss                           | Net revenue gain or loss on your investments.                                              |
| I.1.4.4    | Percentage and Log Returns                | Measuring returns as a simple percent or using logs for better math properties.            |
| I.1.4.5    | Geometric Brownian Motion                | A random walk model often used to simulate stock prices.                                   |
| I.1.4.6    | Discrete and Continuous Compounding       | How interest can be added at intervals or continuously.                                    |
| I.1.4.7    | Period Log Returns in Discrete Time       | Applying log returns for specific intervals.                                               |
| I.1.4.8    | Return on a Linear Portfolio              | Total return from a mix of assets, weighted linearly.                                      |
| I.1.4.9    | Sources of Returns                        | Where gains/losses come from (price changes, dividends, etc.).                             |
| I.1.5      | Functions of Several Variables            | Dealing with multiple inputs (like different factors in a portfolio).                      |
| I.1.5.1    | Partial Derivatives: Function of Two Variables | How a function changes when you tweak one input at a time.                                 |
| I.1.5.2    | Partial Derivatives: Function of Several Variables | Same idea, but with more inputs.                                                          |
| I.1.5.3    | Stationary Points                         | Spots where partial derivatives are zero, potential maxima/minima.                         |
| I.1.5.4    | Optimization                              | Finding the best outcome by adjusting multiple variables (like choosing portfolio weights). |
| I.1.5.5    | Total Derivatives                         | Combine how each variable affects the outcome all at once.                                 |
| I.1.6      | Taylor Expansion                          | Approximate complicated functions by simpler polynomials.                                  |
| I.1.6.1    | Definition and Examples                   | How to expand a function around a point to simplify calculations.                         |
| I.1.6.2    | Risk Factors and Their Sensitivities      | Measuring how small changes in factors (like interest rates) affect returns.               |
| I.1.6.3    | Some Financial Applications of Taylor Expansion | Approximate asset prices or risk quickly.                                               |
| I.1.6.4    | Multivariate Taylor Expansion             | The same but with many variables at once.                                                 |
| I.1.7      | Summary and Conclusions                   | Wraps up how calculus underpins finance formulas and return analysis.                     |
| I.2        | Essential Linear Algebra for Finance      | Using matrices/vectors to handle big sets of numbers (like portfolios).                     |
| I.2.1      | Introduction                              | Overview of why matrices matter in finance (covariance, returns, etc.).                     |
| I.2.2      | Matrix Algebra and its Mathematical Applications | Learning how to add, multiply, and manipulate matrices.                                  |
| I.2.2.1    | Basic Terminology                         | Rows, columns, dimension, etc.                                                              |
| I.2.2.2    | Laws of Matrix Algebra                    | Rules for how matrices combine (like distributive and associative properties).              |
| I.2.2.3    | Singular Matrices                         | Matrices that can’t be inverted (problematic for calculations).                             |
| I.2.2.4    | Determinants                              | Single number that tells you if a matrix is invertible and other properties.                |
| I.2.2.5    | Matrix Inversion                          | How to “divide” with matrices, crucial in portfolio math.                                   |
| I.2.2.6    | Solution of Simultaneous Linear Equations | Using matrix methods to solve many equations at once.                                       |
| I.2.2.7    | Quadratic Forms                           | Expressions like x^T A x, used in measuring portfolio variance.                             |
| I.2.2.8    | Definite Matrices                         | Ensures your quadratic form is always positive or negative (important for risk measures).   |
| I.2.3      | Eigenvectors and Eigenvalues              | Special directions and scaling factors for transformations.                                 |
| I.2.3.1    | Matrices as Linear Transformations        | Seeing matrices as “functions” that stretch/rotate vectors.                                 |
| I.2.3.2    | Formal Definitions                        | The math behind eigen-stuff.                                                               |
| I.2.3.3    | The Characteristic Equation               | Used to find eigenvalues (solutions for the scale factor).                                  |
| I.2.3.4    | Eigenvalues and Eigenvectors of a 2 × 2 Correlation Matrix | A simple example.                                                                         |
| I.2.3.5    | Properties of Eigenvalues and Eigenvectors | Uniqueness, sum, and product relationships.                                               |
| I.2.3.6    | Using Excel to Find Eigenvalues and Eigenvectors | A practical approach for smaller data sets.                                              |
| I.2.3.7    | Eigenvalue Test for Definiteness          | Checking if your matrix is positive-definite (good for variance).                           |
| I.2.4      | Applications to Linear Portfolios         | Using matrix math to describe portfolio risk, return, and correlations.                    |
| I.2.4.1    | Covariance and Correlation Matrices       | Key to measuring how assets move together.                                                 |
| I.2.4.2    | Portfolio Risk and Return in Matrix Notation | A neat formula for your portfolio’s stats.                                              |
| I.2.4.3    | Positive Definiteness of Covariance and Correlation Matrices | Ensures valid risk calculations.                                                         |
| I.2.4.4    | Eigenvalues and Eigenvectors of Covariance and Correlation Matrices | See the main directions of risk.                                                     |
| I.2.5      | Matrix Decomposition                      | Breaking matrices into simpler factors (like factoring numbers).                           |
| I.2.5.1    | Spectral Decomposition of a Symmetric Matrix | Rewriting as a sum of eigenvalues/vectors.                                               |
| I.2.5.2    | Similarity Transforms                     | Reorienting a matrix without changing its core properties.                                 |
| I.2.5.3    | Cholesky Decomposition                   | Fast way to handle covariance matrices (like for simulations).                             |
| I.2.5.4    | LU Decomposition                          | Break a matrix into lower and upper parts for easier solving.                              |
| I.2.6      | Principal Component Analysis              | A method to find the main patterns in big datasets.                                        |
| I.2.6.1    | Definition of Principal Components        | New “axes” that explain most variance in data.                                             |
| I.2.6.2    | Principal Component Representation        | Rewriting data in terms of these key factors.                                              |
| I.2.6.3    | Case Study: PCA of European Equity Indices | Example of how to find main market drivers.                                               |
| I.2.7      | Summary and Conclusions                   | How linear algebra helps build and analyze financial models.                               |
| I.3        | Probability and Statistics                | The math of uncertainty and how to quantify risk.                                          |
| I.3.1      | Introduction                              | Overview of why probability is key in finance (random stock movements, etc.).              |
| I.3.2      | Basic Concepts                            | Classical vs. Bayesian, probability laws, distributions, etc.                              |
| I.3.2.1    | Classical versus Bayesian Approaches      | Different ways to interpret probabilities.                                                |
| I.3.2.2    | Laws of Probability                       | The rules for combining events (AND, OR, independence).                                    |
| I.3.2.3    | Density and Distribution Functions        | How probability is spread over possible outcomes.                                         |
| I.3.2.4    | Samples and Histograms                   | Ways to visualize or approximate real data sets.                                          |
| I.3.2.5    | Expected Value and Sample Mean            | Average outcome of random processes.                                                     |
| I.3.2.6    | Variance                                 | Measure of how spread out the data is.                                                   |
| I.3.2.7    | Skewness and Kurtosis                    | How lopsided or peaked a distribution is.                                                |
| I.3.2.8    | Quantiles, Quartiles, and Percentiles    | Cutoffs that slice data into chunks.                                                     |
| I.3.3      | Univariate Distributions                  | Distribution of a single random variable (like daily returns).                              |
| I.3.3.1    | Binomial Distribution                     | Number of successes in repeated yes/no trials.                                              |
| I.3.3.2    | Poisson and Exponential Distributions     | Modeling rare events or waiting times.                                                     |
| I.3.3.3    | Uniform Distribution                      | All outcomes equally likely.                                                                |
| I.3.3.4    | Normal Distribution                       | The famous bell curve, central in finance.                                                 |
| I.3.3.5    | Lognormal Distribution                    | Distribution of variables whose log is normal (like stock prices).                          |
| I.3.3.6    | Normal Mixture Distributions              | Mixing multiple normals for more flexibility.                                              |
| I.3.3.7    | Student t Distributions                   | Heavier tails than normal, capturing bigger jumps.                                         |
| I.3.3.8    | Sampling Distributions                    | Distribution of sample-based statistics (like sample mean).                                |
| I.3.3.9    | Generalized Extreme Value Distributions   | Focusing on extreme events (crashes).                                                     |
| I.3.3.10   | Generalized Pareto Distribution           | Also focuses on tails/extremes.                                                            |
| I.3.3.11   | Stable Distributions                      | Distributions that remain stable under addition (like Lévy stable).                        |
| I.3.3.12   | Kernels                                   | Smooth functions used for non-parametric density estimation.                               |
| I.3.4      | Multivariate Distributions                | Distributions for multiple variables (like returns of many stocks).                        |
| I.3.4.1    | Bivariate Distributions                   | Simplest multivariate case with two variables.                                             |
| I.3.4.2    | Independent Random Variables              | No connection between outcomes.                                                            |
| I.3.4.3    | Covariance                                | How two variables move together.                                                           |
| I.3.4.4    | Correlation                               | Standardized measure of how two variables move in sync.                                    |
| I.3.4.5    | Multivariate Continuous Distributions     | Continuous probability across multiple dimensions.                                         |
| I.3.4.6    | Multivariate Normal Distributions         | Multi-dimensional bell curves, crucial for portfolio theory.                               |
| I.3.4.7    | Bivariate Normal Mixture Distributions    | Combining multiple 2D normals.                                                            |
| I.3.4.8    | Multivariate Student t Distributions      | Heavier tails in multi-dimensional space.                                                 |
| I.3.5      | Introduction to Statistical Inference     | Using sample data to guess about the real-world population.                                |
| I.3.5.1    | Quantiles, Critical Values, and Confidence Intervals | Ways to test or bound your estimates.                                           |
| I.3.5.2    | Central Limit Theorem                     | Sums of many random variables tend toward a normal distribution.                           |
| I.3.5.3    | Confidence Intervals Based on Student t Distribution | Building intervals when sample size is small.                                     |
| I.3.5.4    | Confidence Intervals for Variance         | Bounding the uncertainty in variance estimates.                                            |
| I.3.5.5    | Hypothesis Tests                          | Deciding if something is statistically significant or just noise.                          |
| I.3.5.6    | Tests on Means                            | Checking if an average is different from a hypothesized value.                             |
| I.3.5.7    | Tests on Variances                        | Checking if a variance matches some reference or changes over time.                        |
| I.3.5.8    | Non-Parametric Tests on Distributions     | Tests that don’t assume normal or any specific shape.                                      |
| I.3.6      | Maximum Likelihood Estimation             | Finding parameter values that make observed data most likely.                              |
| I.3.6.1    | The Likelihood Function                   | Formula for how likely your data is given the parameters.                                  |
| I.3.6.2    | Finding the Maximum Likelihood Estimates  | Picking the parameters that maximize that likelihood.                                      |
| I.3.6.3    | Standard Errors on Mean and Variance Estimates | Measuring how uncertain those estimates are.                                       |
| I.3.7      | Stochastic Processes in Discrete and Continuous Time | Random processes evolving step-by-step or smoothly.                                |
| I.3.7.1    | Stationary and Integrated Processes in Discrete Time | Series that hover around a mean vs. “wandering” series.                           |
| I.3.7.2    | Mean Reverting Processes and Random Walks in Continuous Time | Processes that bounce back vs. drift randomly.                               |
| I.3.7.3    | Stochastic Models for Asset Prices and Returns | Ideas like geometric Brownian motion for stocks.                                  |
| I.3.7.4    | Jumps and the Poisson Process             | Modeling sudden, discrete events.                                                         |
| I.3.8      | Summary and Conclusions                   | Wraps up probability and stats essentials for finance.                                     |
| I.4        | Introduction to Linear Regression         | Using lines to fit data, predict outcomes, and measure relationships.                       |
| I.4.1      | Introduction                              | Big picture of regression and why it’s useful in finance.                                   |
| I.4.2      | Simple Linear Regression                  | One predictor, one outcome, best-fit line approach.                                         |
| I.4.2.1    | Simple Linear Model                       | \( y = a + b \times x \).                                                                   |
| I.4.2.2    | Ordinary Least Squares                    | Method that minimizes the sum of squared errors.                                            |
| I.4.2.3    | Properties of the Error Process           | Assumptions about residuals (mean zero, constant variance).                                 |
| I.4.2.4    | ANOVA and Goodness of Fit                 | Measuring how well the line explains variance in \( y \).                                   |
| I.4.2.5    | Hypothesis Tests on Coefficients          | Checking if slope or intercept differ from zero.                                            |
| I.4.2.6    | Reporting the Estimated Regression Model  | How to show results (equations, \( R^2 \), etc.).                                           |
| I.4.2.7    | Excel Estimation of the Simple Linear Model | Practical approach using Excel’s built-in tools.                                           |
| I.4.3      | Properties of OLS Estimators              | Analyzing the math behind the best-fit line parameters.                                     |
| I.4.3.1    | Estimates and Estimators                  | Realized values vs. the formula that produces them.                                         |
| I.4.3.2    | Unbiasedness and Efficiency               | The estimator hits the true value on average, and with minimal variance.                   |
| I.4.3.3    | Gauss–Markov Theorem                      | OLS is the “best” linear unbiased estimator if assumptions hold.                            |
| I.4.3.4    | Consistency and Normality of OLS Estimators | They converge to the true value over large samples, often normal in distribution.          |
| I.4.3.5    | Testing for Normality                     | Checking if residuals are roughly bell-shaped.                                              |
| I.4.4      | Multivariate Linear Regression            | Multiple \( x \) variables explaining \( y \).                                              |
| I.4.4.1    | Simple Linear Model and OLS in Matrix Notation | Rewriting many \( x \)’s in matrix form.                                                   |
| I.4.4.2    | General Linear Model                      | Flexible version for more complex setups.                                                  |
| I.4.4.3    | Case Study: A Multiple Regression         | Real-world example analyzing multiple predictors.                                           |
| I.4.4.4    | Multiple Regression in Excel              | Steps to do it using data analysis tools.                                                  |
| I.4.4.5    | Hypothesis Testing in Multiple Regression | See which predictors matter.                                                               |
| I.4.4.6    | Testing Multiple Restrictions             | Checking if a group of coefficients is zero.                                               |
| I.4.4.7    | Confidence Intervals                      | Range of values likely to contain the true coefficient.                                     |
| I.4.4.8    | Multicollinearity                         | When predictors overlap, messing up coefficient estimates.                                 |
| I.4.4.9    | Case Study: Determinants of Credit Spreads | Seeing which factors influence bond yield differences.                                      |
| I.4.4.10   | Orthogonal Regression                     | Alternative approach that handles measurement errors differently.                           |
| I.4.5      | Autocorrelation and Heteroscedasticity    | When errors are correlated over time or vary in size.                                       |
| I.4.5.1    | Causes of Autocorrelation and Heteroscedasticity | Reasons why residuals might be related or uneven.                                         |
| I.4.5.2    | Consequences of Autocorrelation and Heteroscedasticity | Standard errors and predictions get skewed.                                                |
| I.4.5.3    | Testing for Autocorrelation               | Durbin-Watson or similar checks.                                                           |
| I.4.5.4    | Testing for Heteroscedasticity            | White test or Breusch-Pagan test.                                                          |
| I.4.5.5    | Generalized Least Squares                 | Fix these issues by adjusting the regression approach.                                     |
| I.4.6      | Applications of Linear Regression in Finance | Ways to use regressions (theory testing, market analysis, etc.).                           |
| I.4.6.1    | Testing a Theory                          | See if data support a finance hypothesis.                                                  |
| I.4.6.2    | Analyzing Empirical Market Behavior       | Measure how variables affect returns.                                                      |
| I.4.6.3    | Optimal Portfolio Allocation              | Use regressions to pick assets.                                                            |
| I.4.6.4    | Regression-Based Hedge Ratios             | Find how much you hedge to offset risk.                                                    |
| I.4.6.5    | Trading on Regression Models              | Building trading signals from fitted relationships.                                        |
| I.4.7      | Summary and Conclusions                   | Final thoughts on using regression tools in finance.                                       |
| I.5        | Numerical Methods in Finance              | Computational tricks to solve finance problems that lack closed-form solutions.            |
| I.5.1      | Introduction                              | What these methods are and why we use them (like iterative solvers).                       |
| I.5.2      | Iteration                                 | Repeating steps to narrow down a solution.                                                 |
| I.5.2.1    | Method of Bisection                       | Simple interval-splitting to find roots.                                                   |
| I.5.2.2    | Newton–Raphson Iteration                 | Faster root-finding that uses derivatives.                                                 |
| I.5.2.3    | Gradient Methods                          | Searching for minima/maxima by following the slope.                                        |
| I.5.3      | Interpolation and Extrapolation           | Estimating values between known data points (or beyond them).                              |
| I.5.3.1    | Linear and Bilinear Interpolation         | Connecting points by straight lines (or on grids).                                         |
| I.5.3.2    | Polynomial Interpolation: Application to Currency Options | Using polynomials to fill in gaps in option pricing data.                                  |
| I.5.3.3    | Cubic Splines: Application to Yield Curves | Smooth fitting of interest rate curves.                                                   |
| I.5.4      | Optimization                              | Finding the best parameters or decisions (like minimal risk or maximum return).            |
| I.5.4.1    | Least Squares Problems                    | Fitting data by minimizing squared errors.                                                 |
| I.5.4.2    | Likelihood Methods                        | Maximizing how probable data are under a chosen model.                                     |
| I.5.4.3    | The EM Algorithm                          | Iterative approach to handle missing data or complex models.                               |
| I.5.4.4    | Case Study: Applying the EM Algorithm to Normal Mixture Densities | Practical example.                                                                        |
| I.5.5      | Finite Difference Approximations          | Approximating derivatives with small steps, used in pricing.                                |
| I.5.5.1    | First and Second Order Finite Differences | Discrete versions of derivatives.                                                           |
| I.5.5.2    | Finite Difference Approximations for the Greeks | Estimate option Greeks numerically.                                                       |
| I.5.5.3    | Finite Difference Solutions to Partial Differential Equations | Numerical approach for PDEs (like Black–Scholes).                                          |
| I.5.6      | Binomial Lattices                         | Step-by-step model for option pricing (up and down moves).                                  |
| I.5.6.1    | Constructing the Lattice                  | Setting up discrete steps for stock price movement.                                         |
| I.5.6.2    | Arbitrage Free Pricing and Risk Neutral Valuation | Fair pricing by eliminating sure-thing profit.                                             |
| I.5.6.3    | Pricing European Options                  | Straightforward binomial tree approach.                                                     |
| I.5.6.4    | Lognormal Asset Price Distributions       | Linking the binomial steps to continuous lognormal shapes.                                  |
| I.5.6.5    | Pricing American Options                  | Add ability to exercise before expiration.                                                 |
| I.5.7      | Monte Carlo Simulation                    | Using random draws to approximate outcomes.                                                |
| I.5.7.1    | Random Numbers                            | Generating pseudo-random sequences for simulations.                                        |
| I.5.7.2    | Simulations from an Empirical or a Given Distribution | Sampling real or theoretical distributions.                                               |
| I.5.7.3    | Case Study: Generating Time Series of Lognormal Asset Prices | Example of how to simulate stock paths.                                                  |
| I.5.7.4    | Simulations on a System of Two Correlated Normal Returns | Capturing correlation between assets.                                                    |
| I.5.7.5    | Multivariate Normal and Student t Distributed Simulations | Handle multi-asset portfolios with thick tails.                                          |
| I.5.8      | Summary and Conclusions                   | How numeric methods help solve tough finance models quickly.                                |
| I.6        | Introduction to Portfolio Theory          | How to choose or mix assets to balance risk and return.                                     |
| I.6.1      | Introduction                              | Quick overview of portfolio theory goals.                                                  |
| I.6.2      | Utility Theory                            | How investors weigh risk vs. reward in a math formula.                                      |
| I.6.2.1    | Properties of Utility Functions           | Shape indicates risk preference (risk-averse, neutral, or seeking).                        |
| I.6.2.2    | Risk Preference                           | How you feel about potential losses vs. gains.                                             |
| I.6.2.3    | How to Determine the Risk Tolerance of an Investor | Measure how comfortable they are with uncertainty.                                       |
| I.6.2.4    | Coefficients of Risk Aversion             | How strongly someone hates risk in numeric form.                                           |
| I.6.2.5    | Some Standard Utility Functions           | Typical forms used in finance (e.g., exponential, power).                                  |
| I.6.2.6    | Mean–Variance Criterion                   | Evaluating investments by average return vs. variance of returns.                          |
| I.6.2.7    | Extension of the Mean–Variance Criterion to Higher Moments | Factoring skew/kurtosis in decision-making.                                               |
| I.6.3      | Portfolio Allocation                      | Dividing your money across assets to optimize risk/return.                                 |
| I.6.3.1    | Portfolio Diversification                 | Mixing uncorrelated assets to reduce risk.                                                 |
| I.6.3.2    | Minimum Variance Portfolios               | Combos that minimize risk for a given return.                                              |
| I.6.3.3    | The Markowitz Problem                     | Classic formula for picking an optimal portfolio.                                          |
| I.6.3.4    | Minimum Variance Portfolios with Many Constraints | Realistic versions of Markowitz with extra conditions.                                    |
| I.6.3.5    | Efficient Frontier                        | The set of best possible trade-offs between risk and return.                               |
| I.6.3.6    | Optimal Allocations Theory of Asset Pricing | Linking these ideas to asset pricing models.                                              |
| I.6.4      | Capital Market Line                       | A line of efficient portfolios when you include a risk-free asset.                         |
| I.6.4.1    | Capital Asset Pricing Model (CAPM)        | Connects an asset’s risk to its expected return via beta.                                  |
| I.6.4.2    | Security Market Line                      | A graph of CAPM’s risk-return relationship.                                               |
| I.6.4.3    | Testing the CAPM                          | Seeing if real data fits that linear relationship.                                         |
| I.6.4.4    | Extensions to CAPM Risk Adjusted Performance Measures | Expansions on the basic CAPM for performance tracking.                                    |
| I.6.5      | CAPM RAPMs                                | Risk-adjusted performance measures that come from CAPM logic.                              |
| I.6.5.1    | Making Decisions Using the Sharpe Ratio   | A simple ratio of return over volatility.                                                 |
| I.6.5.2    | Adjusting the Sharpe Ratio for Autocorrelation | Fix for when returns aren’t independent.                                                |
| I.6.5.3    | Adjusting the Sharpe Ratio for Higher Moments | Fix for skew/kurtosis.                                                                   |
| I.6.5.4    | Generalized Sharpe Ratio                  | More flexible version of the Sharpe ratio.                                                |
| I.6.5.5    | Kappa Indices, Omega, and Sortino Ratio   | Other ways to measure risk-adjusted returns.                                              |
| I.6.5.6    | Summary and Conclusions                   | Big picture of portfolio selection, CAPM, and performance metrics.                        |
| II.1       | Factor Models                              | Explaining returns with underlying risk “factors.”                                          |
| II.1.1     | Introduction                              | What factor models are and why they matter (common drivers of asset returns).               |
| II.1.2     | Single Factor Models                      | Using just one main driver of returns (like a market index).                                |
| II.1.2.1   | Single Index Model                        | Often the market index is that factor.                                                      |
| II.1.2.2   | Estimating Portfolio Characteristics using OLS | Regress portfolio returns on the factor.                                                 |
| II.1.2.3   | Estimating Portfolio Risk using EWMA      | Measure changing volatility over time.                                                      |
| II.1.2.4   | Relationship between Beta, Correlation, and Relative Volatility | How beta links to correlation and risk.                                                  |
| II.1.2.5   | Risk Decomposition in a Single Factor Model | Splitting risk into factor and specific parts.                                             |
| II.1.3     | Multi-Factor Models                       | More than one factor (e.g., interest rates, market index, sector).                         |
| II.1.3.1   | Multi-factor Models of Asset or Portfolio Returns | Multiple economic or style factors.                                                     |
| II.1.3.2   | Style Attribution Analysis                | See how much returns come from styles (value, growth, etc.).                                |
| II.1.3.3   | General Formulation of Multi-factor Model | The math for multiple factors.                                                             |
| II.1.3.4   | Multi-factor Models of International Portfolios | Capturing region or currency factors.                                                    |
| II.1.4     | Case Study: Estimation of Fundamental Factor Models | Example using fundamental data (like P/E ratio).                                         |
| II.1.4.1   | Estimating Systematic Risk for a Portfolio of US Stocks | Measure betas for many stocks.                                                           |
| II.1.4.2   | Multicollinearity: A Problem with Fundamental Factor Models | Overlapping factors.                                                                     |
| II.1.4.3   | Estimating Fundamental Factor Models by Orthogonal Regression | Fix multicollinearity by rotating factors.                                              |
| II.1.5     | Analysis of Barra Model                   | A well-known commercial factor model.                                                      |
| II.1.5.1   | Risk Indices, Descriptors, and Fundamental Betas | Building blocks in Barra.                                                               |
| II.1.5.2   | Model Specification and Risk Decomposition | How Barra breaks down total risk.                                                        |
| II.1.6     | Tracking Error and Active Risk            | Measuring how a portfolio deviates from a benchmark.                                        |
| II.1.6.1   | Ex Post versus Ex Ante Measurement of Risk and Return | Historical vs. predicted measures.                                                     |
| II.1.6.2   | Definition of Active Returns              | Difference between portfolio and benchmark returns.                                        |
| II.1.6.3   | Definition of Active Weights              | How portfolio weighting differs from benchmark weighting.                                  |
| II.1.6.4   | Ex Post Tracking Error                   | Realized difference from benchmark over past data.                                         |
| II.1.6.5   | Ex Post Mean-Adjusted Tracking Error      | Adjusting for average differences.                                                        |
| II.1.6.6   | Ex Ante Tracking Error                   | Predicted future deviation from benchmark.                                                |
| II.1.6.7   | Ex Ante Mean-Adjusted Tracking Error      | Predicted difference factoring in average shift.                                           |
| II.1.6.8   | Clarification of the Definition of Active Risk | All the ways to measure how “off-benchmark” you are.                                     |
| II.1.7     | Summary and Conclusions                   | Factor models let you decompose and manage the big drivers of risk and returns.            |
| II.2       | Principal Component Analysis              | Finding the key directions of variation in data sets (like interest rates).                |
| II.2.1     | Introduction                              | How PCA reveals patterns in big correlated data sets.                                      |
| II.2.2     | Review of Principal Component Analysis    | Quick refresher on PCA ideas.                                                             |
| II.2.2.1   | Definition of Principal Components        | New axes that capture the most variance.                                                  |
| II.2.2.2   | Principal Component Representation        | Rewriting data using fewer, powerful components.                                          |
| II.2.2.3   | Frequently Asked Questions                | Clarifications on how PCA works or is interpreted.                                         |
| II.2.3     | Case Study: PCA of UK Government Yield Curves | See main movements in interest rates.                                                   |
| II.2.3.1   | Properties of UK Interest Rates           | Overview of the data.                                                                     |
| II.2.3.2   | Volatility and Correlation of UK Spot Rates | How interest rates fluctuate together.                                                  |
| II.2.3.3   | PCA on UK Spot Rates Correlation Matrix   | Standard approach to find main factors.                                                  |
| II.2.3.4   | Principal Component Representation        | Show data in terms of the first few principal components.                                 |
| II.2.3.5   | PCA on UK Short Spot Rates Covariance Matrix | Same process but on covariance.                                                        |
| II.2.4     | Term Structure Factor Models              | Modeling interest rate changes across different maturities.                              |
| II.2.4.1   | Interest Rate Sensitive Portfolios        | Focusing on how bond prices shift.                                                       |
| II.2.4.2   | Factor Models for Currency Forward Positions | Analyzing key movements in FX markets.                                                 |
| II.2.4.3   | Factor Models for Commodity Futures Portfolios | Capturing broad commodity risk factors.                                                |
| II.2.4.4   | Application to Portfolio Immunization     | Hedging interest rate risk systematically.                                               |
| II.2.4.5   | Application to Asset–Liability Management | Matching assets to future liabilities.                                                  |
| II.2.4.6   | Application to Portfolio Risk Measurement | Quickly see big risk exposures.                                                         |
| II.2.4.7   | Multiple Curve Factor Models              | Extending to more complex yield curves.                                                 |
| II.2.5     | Equity PCA Factor Models                  | Using PCA for stocks, picking out main market drivers.                                   |
| II.2.5.1   | Model Structure                           | How to set up PCA for equities.                                                          |
| II.2.5.2   | Specific Risks and Dimension Reduction    | Separate common factors from idiosyncratic risk.                                         |
| II.2.5.3   | Case Study: PCA Factor Model for DJIA Portfolios | See how Dow stocks move in sync or differ.                                            |
| II.2.6     | Summary and Conclusions                   | PCA is a powerful tool to simplify big correlated data sets.                              |


Continuing the Markdown table:

```markdown
| Section    | Title                                      | Description                                                                                  |
|------------|--------------------------------------------|----------------------------------------------------------------------------------------------|
| II.1       | Factor Models                              | Explaining returns with underlying risk “factors.”                                          |
| II.1.1     | Introduction                              | What factor models are and why they matter (common drivers of asset returns).               |
| II.1.2     | Single Factor Models                      | Using just one main driver of returns (like a market index).                                |
| II.1.2.1   | Single Index Model                        | Often the market index is that factor.                                                      |
| II.1.2.2   | Estimating Portfolio Characteristics using OLS | Regress portfolio returns on the factor.                                                 |
| II.1.2.3   | Estimating Portfolio Risk using EWMA      | Measure changing volatility over time.                                                      |
| II.1.2.4   | Relationship between Beta, Correlation, and Relative Volatility | How beta links to correlation and risk.                                                  |
| II.1.2.5   | Risk Decomposition in a Single Factor Model | Splitting risk into factor and specific parts.                                             |
| II.1.3     | Multi-Factor Models                       | More than one factor (e.g., interest rates, market index, sector).                         |
| II.1.3.1   | Multi-factor Models of Asset or Portfolio Returns | Multiple economic or style factors.                                                     |
| II.1.3.2   | Style Attribution Analysis                | See how much returns come from styles (value, growth, etc.).                                |
| II.1.3.3   | General Formulation of Multi-factor Model | The math for multiple factors.                                                             |
| II.1.3.4   | Multi-factor Models of International Portfolios | Capturing region or currency factors.                                                    |
| II.1.4     | Case Study: Estimation of Fundamental Factor Models | Example using fundamental data (like P/E ratio).                                         |
| II.1.4.1   | Estimating Systematic Risk for a Portfolio of US Stocks | Measure betas for many stocks.                                                           |
| II.1.4.2   | Multicollinearity: A Problem with Fundamental Factor Models | Overlapping factors.                                                                     |
| II.1.4.3   | Estimating Fundamental Factor Models by Orthogonal Regression | Fix multicollinearity by rotating factors.                                              |
| II.1.5     | Analysis of Barra Model                   | A well-known commercial factor model.                                                      |
| II.1.5.1   | Risk Indices, Descriptors, and Fundamental Betas | Building blocks in Barra.                                                               |
| II.1.5.2   | Model Specification and Risk Decomposition | How Barra breaks down total risk.                                                        |
| II.1.6     | Tracking Error and Active Risk            | Measuring how a portfolio deviates from a benchmark.                                        |
| II.1.6.1   | Ex Post versus Ex Ante Measurement of Risk and Return | Historical vs. predicted measures.                                                     |
| II.1.6.2   | Definition of Active Returns              | Difference between portfolio and benchmark returns.                                        |
| II.1.6.3   | Definition of Active Weights              | How portfolio weighting differs from benchmark weighting.                                  |
| II.1.6.4   | Ex Post Tracking Error                   | Realized difference from benchmark over past data.                                         |
| II.1.6.5   | Ex Post Mean-Adjusted Tracking Error      | Adjusting for average differences.                                                        |
| II.1.6.6   | Ex Ante Tracking Error                   | Predicted future deviation from benchmark.                                                |
| II.1.6.7   | Ex Ante Mean-Adjusted Tracking Error      | Predicted difference factoring in average shift.                                           |
| II.1.6.8   | Clarification of the Definition of Active Risk | All the ways to measure how “off-benchmark” you are.                                     |
| II.1.7     | Summary and Conclusions                   | Factor models let you decompose and manage the big drivers of risk and returns.            |
| II.2       | Principal Component Analysis              | Finding the key directions of variation in data sets (like interest rates).                |
| II.2.1     | Introduction                              | How PCA reveals patterns in big correlated data sets.                                      |
| II.2.2     | Review of Principal Component Analysis    | Quick refresher on PCA ideas.                                                             |
| II.2.2.1   | Definition of Principal Components        | New axes that capture the most variance.                                                  |
| II.2.2.2   | Principal Component Representation        | Rewriting data using fewer, powerful components.                                          |
| II.2.2.3   | Frequently Asked Questions                | Clarifications on how PCA works or is interpreted.                                         |
| II.2.3     | Case Study: PCA of UK Government Yield Curves | See main movements in interest rates.                                                   |
| II.2.3.1   | Properties of UK Interest Rates           | Overview of the data.                                                                     |
| II.2.3.2   | Volatility and Correlation of UK Spot Rates | How interest rates fluctuate together.                                                  |
| II.2.3.3   | PCA on UK Spot Rates Correlation Matrix   | Standard approach to find main factors.                                                  |
| II.2.3.4   | Principal Component Representation        | Show data in terms of the first few principal components.                                 |
| II.2.3.5   | PCA on UK Short Spot Rates Covariance Matrix | Same process but on covariance.                                                        |
| II.2.4     | Term Structure Factor Models              | Modeling interest rate changes across different maturities.                              |
| II.2.4.1   | Interest Rate Sensitive Portfolios        | Focusing on how bond prices shift.                                                       |
| II.2.4.2   | Factor Models for Currency Forward Positions | Analyzing key movements in FX markets.                                                 |
| II.2.4.3   | Factor Models for Commodity Futures Portfolios | Capturing broad commodity risk factors.                                                |
| II.2.4.4   | Application to Portfolio Immunization     | Hedging interest rate risk systematically.                                               |
| II.2.4.5   | Application to Asset–Liability Management | Matching assets to future liabilities.                                                  |
| II.2.4.6   | Application to Portfolio Risk Measurement | Quickly see big risk exposures.                                                         |
| II.2.4.7   | Multiple Curve Factor Models              | Extending to more complex yield curves.                                                 |
| II.2.5     | Equity PCA Factor Models                  | Using PCA for stocks, picking out main market drivers.                                   |
| II.2.5.1   | Model Structure                           | How to set up PCA for equities.                                                          |
| II.2.5.2   | Specific Risks and Dimension Reduction    | Separate common factors from idiosyncratic risk.                                         |
| II.2.5.3   | Case Study: PCA Factor Model for DJIA Portfolios | See how Dow stocks move in sync or differ.                                            |
| II.2.6     | Summary and Conclusions                   | PCA is a powerful tool to simplify big correlated data sets.                              |
| II.3       | Classical Models of Volatility and Correlation | Measuring how returns vary and move together over time.                                     |
| II.3.1     | Introduction                              | Overview of classical approaches to track changing risk.                                    |
| II.3.2     | Variance and Volatility                   | Ways to measure the wiggle in returns.                                                      |
| II.3.2.1   | Volatility and the Square-Root-of-Time Rule | Scaling daily volatility to longer horizons.                                               |
| II.3.2.2   | Constant Volatility Assumption            | Simplest assumption (like old Black–Scholes).                                              |
| II.3.2.3   | Volatility when Returns are Autocorrelated | Deals with correlated returns over time.                                                   |
| II.3.2.4   | Remarks about Volatility                  | Practical insights on how volatility really behaves.                                        |
| II.3.3     | Covariance and Correlation                | How pairs of assets move together.                                                         |
| II.3.3.1   | Definition of Covariance and Correlation  | Measure the strength of co-movement.                                                       |
| II.3.3.2   | Correlation Pitfalls                      | Correlation can be misleading in certain situations.                                        |
| II.3.3.3   | Covariance Matrices                       | Collecting all pairwise covariances in one matrix.                                          |
| II.3.3.4   | Scaling Covariance Matrices               | How to adjust them for different time horizons.                                             |
| II.3.4     | Equally Weighted Averages                 | Simple way to estimate volatility by averaging squares of returns.                         |
| II.3.4.1   | Unconditional Variance and Volatility     | Straightforward, ignoring any time pattern.                                                |
| II.3.4.2   | Unconditional Covariance and Correlation  | Also ignoring time dynamics.                                                               |
| II.3.4.3   | Forecasting with Equally Weighted Averages | Not always great, but easy to do.                                                         |
| II.3.5     | Precision of Equally Weighted Estimates   | How uncertain they can be.                                                                 |
| II.3.5.1   | Confidence Intervals for Variance and Volatility | Bounding your volatility estimates.                                                     |
| II.3.5.2   | Standard Error of Variance Estimator      | A measure of how precise your variance guess is.                                           |
| II.3.5.3   | Standard Error of Volatility Estimator    | Same idea, but for volatility.                                                             |
| II.3.5.4   | Standard Error of Correlation Estimator   | How certain you are about correlation guesses.                                             |
| II.3.6     | Case Study: Volatility and Correlation of US Treasuries | Real example on interest rate moves.                                                   |
| II.3.6.1   | Choosing the Data                         | Picking the right yield data, time intervals.                                              |
| II.3.6.2   | Our Data                                  | Example data set used in the study.                                                        |
| II.3.6.3   | Effect of Sample Period                   | Volatility estimates can change if you pick different dates.                               |
| II.3.6.4   | How to Calculate Changes in Interest Rates | Differences, log changes, etc.                                                            |
| II.3.7     | Equally Weighted Moving Averages          | Method that uses a rolling window approach.                                                |
| II.3.7.1   | Effect of Volatility Clusters             | Big moves often follow big moves.                                                          |
| II.3.7.2   | Pitfalls of the Equally Weighted Moving Average Method | All data in the window weighted equally, ignoring recency.                                |
| II.3.7.3   | Three Ways to Forecast Long Term Volatility | From simple average to more complex models.                                              |
| II.3.8     | Exponentially Weighted Moving Averages    | Weights recent data more than old data.                                                    |
| II.3.8.1   | Statistical Methodology                   | How the exponentially decaying factor is applied.                                          |
| II.3.8.2   | Interpretation of Lambda                  | Parameter that decides how fast past data “fades away.”                                    |
| II.3.8.3   | Properties of EWMA Estimators             | Can adapt quickly to new volatility.                                                      |
| II.3.8.4   | Forecasting with EWMA                     | Produce short-term volatility forecasts.                                                  |
| II.3.8.5   | Standard Errors for EWMA Forecasts        | Measuring the uncertainty in those forecasts.                                             |
| II.3.8.6   | RiskMetricsTM Methodology                | JP Morgan’s well-known EWMA-based approach.                                              |
| II.3.8.7   | Orthogonal EWMA versus RiskMetrics EWMA   | A variant that decomposes risk factors differently.                                        |
| II.3.9     | Summary and Conclusions                   | Classical ways to measure risk, from simple to more dynamic methods.                      |
| II.4       | Introduction to GARCH Models              | Advanced volatility models that let volatility evolve over time.                            |
| II.4.1     | Introduction                              | Why we need GARCH for better volatility forecasts.                                          |
| II.4.2     | The Symmetric Normal GARCH Model          | Base model for conditional volatility.                                                      |
| II.4.2.1   | Model Specification                       | Formula that updates variance based on past errors.                                         |
| II.4.2.2   | Parameter Estimation                      | How to fit the model to historical data.                                                    |
| II.4.2.3   | Volatility Estimates                      | Time-varying volatility outputs.                                                            |
| II.4.2.4   | GARCH Volatility Forecasts                | Projecting volatility forward.                                                              |
| II.4.2.5   | Imposing Long Term Volatility             | Ensuring the model converges to a long-run average.                                         |
| II.4.2.6   | Comparison of GARCH and EWMA Volatility Models | Which does better under different conditions.                                             |
| II.4.3     | Asymmetric GARCH Models                   | Letting volatility respond differently to positive/negative shocks.                        |
| II.4.3.1   | A-GARCH                                   | One type of asymmetry.                                                                     |
| II.4.3.2   | GJR-GARCH                                 | Another popular asymmetry approach.                                                        |
| II.4.3.3   | Exponential GARCH                         | Logs the variance to keep it always positive.                                              |
| II.4.3.4   | Analytic E-GARCH Volatility Term Structure Forecasts | Projecting future volatility with E-GARCH formula.                                       |
| II.4.3.5   | Volatility Feedback                       | How changing volatility can affect prices.                                                 |
| II.4.4     | Non-Normal GARCH Models                   | Allowing returns to have heavier tails.                                                    |
| II.4.4.1   | Student t GARCH Models                    | Use t-distribution for fatter tails.                                                       |
| II.4.4.2   | Case Study: Comparison of GARCH Models for the FTSE 100 | Which GARCH type fits real data better.                                                  |
| II.4.4.3   | Normal Mixture GARCH Models               | Mixing multiple normals for more flexible shape.                                           |
| II.4.4.4   | Markov Switching GARCH                    | Volatility regime switches over time.                                                     |
| II.4.5     | GARCH Covariance Matrices                 | Extends GARCH to multiple assets simultaneously.                                           |
| II.4.5.1   | Estimation of Multivariate GARCH Models   | More complicated parameter fitting.                                                        |
| II.4.5.2   | Constant and Dynamic Conditional Correlation GARCH | How correlations move over time.                                                         |
| II.4.5.3   | Factor GARCH                              | Factor-based version to handle many assets.                                                |
| II.4.6     | Orthogonal GARCH                          | Diagonalizing the covariance to simplify the model.                                        |
| II.4.6.1   | Model Specification                       | How it’s set up.                                                                           |
| II.4.6.2   | Case Study: A Comparison of RiskMetrics and O-GARCH | See how each approach performs.                                                          |
| II.4.6.3   | Splicing Methods for Constructing Large Covariance Matrices | Combining different pieces for a giant portfolio.                                       |
| II.4.7     | Monte Carlo Simulation with GARCH Models  | Simulating future returns with time-varying volatility.                                   |
| II.4.7.1   | Simulation with Volatility Clustering     | Capturing clusters of big moves.                                                          |
| II.4.7.2   | Simulation with Volatility Clustering Regimes | Jumping between calm and turbulent markets.                                              |
| II.4.7.3   | Simulation with Correlation Clustering    | Times when assets move together more closely.                                             |
| II.4.8     | Applications of GARCH Models              | Practical uses like option pricing, VaR, portfolio optimization.                          |
| II.4.8.1   | Option Pricing with GARCH Diffusions      | More realistic than constant volatility.                                                  |
| II.4.8.2   | Pricing Path-Dependent European Options   | Handle entire price paths with GARCH.                                                     |
| II.4.8.3   | Value-at-Risk Measurement                 | Capturing dynamic risk over time.                                                         |
| II.4.8.4   | Estimation of Time Varying Sensitivities  | How betas or exposures change day to day.                                                 |
| II.4.8.5   | Portfolio Optimization                    | Factoring in changing volatility to choose weights.                                       |
| II.4.9     | Summary and Conclusions                   | GARCH is a flexible tool for modeling changing volatility.                                |
| II.5       | Time Series Models and Cointegration      | Analyzing how variables evolve over time and might share trends.                            |
| II.5.1     | Introduction                              | Why time-series methods matter in finance (predicting price changes).                       |
| II.5.2     | Stationary Processes                      | Processes with constant mean/variance over time.                                            |
| II.5.2.1   | Time Series Models                        | AR, MA, ARMA, etc.                                                                          |
| II.5.2.2   | Inversion and the Lag Operator            | Ways to manipulate these models mathematically.                                             |
| II.5.2.3   | Response to Shocks                        | How processes react to big news.                                                            |
| II.5.2.4   | Estimation                                | Fitting ARMA-type models.                                                                   |
| II.5.2.5   | Prediction                                | Forecasting future values.                                                                  |
| II.5.2.6   | Multivariate Models for Stationary Processes | Extends ARMA ideas to multiple series.                                                    |
| II.5.3     | Stochastic Trends                         | Series that drift without returning to a stable mean (random walks).                        |
| II.5.3.1   | Random Walks and Efficient Markets        | Popular model for stock prices.                                                            |
| II.5.3.2   | Integrated Processes and Stochastic Trends | Differencing needed to find stationarity.                                                  |
| II.5.3.3   | Deterministic Trends                      | Time-based predictable patterns.                                                           |
| II.5.3.4   | Unit Root Tests                           | Check if a series is a non-stationary random walk.                                         |
| II.5.3.5   | Unit Roots in Asset Prices                | Often found in stock prices.                                                               |
| II.5.3.6   | Unit Roots in Interest Rates, Credit Spreads, and Implied Volatility | Common in finance data.                                                               |
| II.5.3.7   | Reconciliation of Time Series and Continuous Time Models | Bridging discrete vs. continuous approaches.                                            |
| II.5.3.8   | Unit Roots in Commodity Prices            | See if they follow random walks.                                                           |
| II.5.4     | Long Term Equilibrium                     | Whether series move together in the long run.                                              |
| II.5.4.1   | Cointegration and Correlation Compared    | Cointegration is about shared trends, correlation is about linear moves.                   |
| II.5.4.2   | Common Stochastic Trends                  | Multiple series drifting together.                                                         |
| II.5.4.3   | Formal Definition of Cointegration        | Existence of a stable combo of non-stationary series.                                       |
| II.5.4.4   | Evidence of Cointegration in Financial Markets | Some markets do move together over time.                                                 |
| II.5.4.5   | Estimation and Testing in Cointegrated Systems | Special methods (Engle-Granger, Johansen).                                              |
| II.5.4.6   | Application to Benchmark Tracking         | Replicate an index by matching that cointegrated relationship.                             |
| II.5.4.7   | Case Study: Cointegration Index Tracking in the Dow Jones Index | Example.                                                                                 |
| II.5.5     | Modeling Short Term Dynamics              | Error correction, Granger causality, etc.                                                 |
| II.5.5.1   | Error Correction Models                   | Bringing series back to equilibrium if they deviate.                                       |
| II.5.5.2   | Granger Causality                         | Seeing if one series helps predict another.                                               |
| II.5.5.3   | Case Study: Pairs Trading Volatility Index Futures | Trade on cointegrated pairs.                                                             |
| II.5.6     | Summary and Conclusions                   | Time series approaches for both short-run and long-run relationships.                     |
| II.6       | Introduction to Copulas                   | A way to model how variables move together beyond just simple correlation.                  |
| II.6.1     | Introduction                              | Overview of copulas, which link individual distributions to a joint distribution.           |
| II.6.2     | Concordance Metrics                       | Measure how well variables “move in the same direction.”                                    |
| II.6.2.1   | Concordance                               | Consistency in ordering between variables.                                                  |
| II.6.2.2   | Rank Correlations                         | Like Spearman’s rho or Kendall’s tau, focusing on relative ranks.                           |
| II.6.3     | Copulas and Associated Theoretical Concepts | The math behind separating marginal distributions from dependence structure.               |
| II.6.3.1   | Simulation of a Single Random Variable    | Starting point before linking them.                                                        |
| II.6.3.2   | Definition of a Copula                    | Function that ties separate distributions together.                                         |
| II.6.3.3   | Conditional Copula Distributions and their Quantile Curves | How to define dependence for each sub-range.                                           |
| II.6.3.4   | Tail Dependence                           | How strongly extremes move together.                                                       |
| II.6.3.5   | Bounds for Dependence                     | Theoretical limits on how variables can co-move.                                            |
| II.6.4     | Examples of Copulas                       | Different shapes for dependence structures.                                                |
| II.6.4.1   | Normal or Gaussian Copulas               | Symmetrical dependence, widely used.                                                       |
| II.6.4.2   | Student t Copulas                        | Heavier tail dependence.                                                                   |
| II.6.4.3   | Normal Mixture Copulas                   | Combine multiple copulas for more complex structures.                                       |
| II.6.4.4   | Archimedean Copulas                      | Simpler formulas for certain shapes of dependence.                                          |
| II.6.5     | Conditional Copula Distributions and Quantile Curves | How each copula handles conditional relationships.                                       |
| II.6.5.1   | Normal or Gaussian Copulas               | A standard baseline.                                                                       |
| II.6.5.2   | Student t Copulas                        | Captures extreme co-movements better.                                                      |
| II.6.5.3   | Normal Mixture Copulas                   | Flexible approach mixing multiple normals.                                                 |
| II.6.5.4   | Archimedean Copulas                      | Sometimes easier to parameterize for certain data.                                         |
| II.6.5.5   | Examples                                 | Quick demos for each.                                                                      |
| II.6.6     | Calibrating Copulas                      | Matching them to real data.                                                                |
| II.6.6.1   | Correspondence between Copulas and Rank Correlations | A direct link to measure dependence.                                                   |
| II.6.6.2   | Maximum Likelihood Estimation            | Find the copula parameters that best fit observed data.                                     |
| II.6.6.3   | How to Choose the Best Copula            | Compare fit, tail dependence, etc.                                                        |
| II.6.7     | Simulation with Copulas                  | Generating correlated random variables with flexible dependence.                           |
| II.6.7.1   | Using Conditional Copulas for Simulation | Dynamically adapt dependence.                                                             |
| II.6.7.2   | Simulation from Elliptical Copulas       | Normal or t distribution shapes.                                                          |
| II.6.7.3   | Simulation with Normal and Student t Copulas | Typical approach to risk modeling.                                                       |
| II.6.7.4   | Simulation from Archimedean Copulas      | Alternative formula style.                                                                |
| II.6.8     | Market Risk Applications                 | Advanced ways to compute VaR, diversification, and optimize portfolios.                   |
| II.6.8.1   | Value-at-Risk Estimation                 | Capturing non-linear co-movements in the tail.                                            |
| II.6.8.2   | Aggregation and Portfolio Diversification | Combining risks with flexible dependence.                                                |
| II.6.8.3   | Using Copulas for Portfolio Optimization | Better handle extreme co-movements.                                                      |
| II.6.9     | Summary and Conclusions                  | Copulas allow more nuanced modeling of joint risk than correlation alone.                 |
| II.7       | Advanced Econometric Models               | Next-level tools for analyzing complex financial data.                                      |
| II.7.1     | Introduction                              | An overview of advanced methods beyond basic regression.                                    |
| II.7.2     | Quantile Regression                       | Modeling different parts (quantiles) of the distribution, not just the mean.               |
| II.7.2.1   | Review of Standard Regression             | Linear, OLS recap.                                                                          |
| II.7.2.2   | What is Quantile Regression?              | Focusing on medians or other percentiles for robust analysis.                              |
| II.7.2.3   | Parameter Estimation in Quantile Regression | Finding lines that minimize absolute errors at each quantile.                             |
| II.7.2.4   | Inference on Linear Quantile Regressions  | Tests/confidence intervals for quantile slopes.                                            |
| II.7.2.5   | Using Copulas for Non-linear Quantile Regression | Combining copulas with quantiles for flexible shapes.                                    |
| II.7.3     | Case Studies in Quantile Regression       | Practical examples of advanced applications.                                               |
| II.7.3.1   | Case Study 1: Quantile Regression of VFTSE on FTSE 100 Index | Seeing how volatility index behaves across conditions.                                  |
| II.7.3.2   | Case Study 2: Hedging with Copula Quantile Regression | Example of advanced hedge approach.                                                     |
| II.7.4     | Other Non-Linear Regression Models        | Expanding beyond basic regression.                                                        |
| II.7.4.1   | Non-linear Least Squares                  | Fitting curves instead of straight lines.                                                 |
| II.7.4.2   | Discrete Choice Models                    | Logistic or probit for yes/no outcomes.                                                   |
| II.7.5     | Structural Breaks and Model Specification | Advanced tools to handle changing relationships over time.                                |
| II.7.5.1   | Testing for Structural Breaks             | Checking if your model changes at certain points in time.                                 |
| II.7.5.2   | Model Specification                       | Choosing the right form for your data.                                                    |
| II.7.6     | High-Frequency Data Models                | Specialized models for tick-level or very frequent data.                                  |
| II.7.6.1   | Data Sources and Filtering                | Handling tick-by-tick or very frequent data.                                              |
| II.7.6.2   | Modeling the Time Between Trades          | Capturing irregular transaction times.                                                    |
| II.7.6.3   | Forecasting Volatility                    | Specialized methods for high-frequency data.                                              |
| II.7.7     | Financial Applications and Software       | Tools for advanced modeling (like EViews, R, Python libs).                                |
| II.7.8     | Summary and Conclusions                   | Advanced methods can handle complexity in real financial markets.                         |
| II.8       | Forecasting and Model Evaluation          | How to see if your model predictions are good and robust.                                    |
| II.8.1     | Introduction                              | Why we need to test our models’ predictive power.                                             |
| II.8.2     | Returns Models                            | Measuring fit and forecasting ability of return predictions.                                  |
| II.8.2.1   | Goodness of Fit                           | Does the model capture the data?                                                            |
| II.8.2.2   | Forecasting                               | Checking how well it predicts out-of-sample.                                                 |
| II.8.2.3   | Simulating Critical Values for Test Statistics | Using simulations to define thresholds for acceptance/rejection.                          |
| II.8.2.4   | Specification Tests for Regime Switching Models | See if your model needs multiple “states.”                                                 |
| II.8.3     | Volatility Models                         | See how well GARCH or others match real market turbulence.                                   |
| II.8.3.1   | Goodness of Fit of GARCH Models           | Do actual volatilities line up with forecasts?                                               |
| II.8.3.2   | Forecasting with GARCH Volatility Models  | Testing how accurate predictions are.                                                       |
| II.8.3.3   | Moving Average Models                    | Simpler volatility approach to compare against.                                              |
| II.8.4     | Forecasting the Tails of a Distribution   | We care about extremes (risk management).                                                   |
| II.8.4.1   | Confidence Intervals for Quantiles       | Bounding worst-case scenarios.                                                             |
| II.8.4.2   | Coverage Tests                           | Checking if your VaR (Value-at-Risk) is correct often enough.                                |
| II.8.4.3   | Application of Coverage Tests to GARCH Models | Validating GARCH-based VaR.                                                              |
| II.8.4.4   | Forecasting Conditional Correlations      | How correlated assets become in certain conditions.                                          |
| II.8.5     | Operational Evaluation                    | See how the model works in real trading or risk management.                                 |
| II.8.5.1   | General Backtesting Algorithm             | Test model predictions on historical data.                                                 |
| II.8.5.2   | Alpha Models                             | Check if the model can generate excess returns.                                             |
| II.8.5.3   | Portfolio Optimization                    | See if the chosen weights truly reduce risk or boost returns.                               |
| II.8.5.4   | Hedging with Futures                     | Test how well hedges hold up in real conditions.                                            |
| II.8.5.5   | Value-at-Risk Measurement                | Main risk measure tested in practice.                                                      |
| II.8.5.6   | Trading Implied Volatility               | See if you can profit from differences in implied vs. realized vol.                         |
| II.8.5.7   | Trading Realized Volatility              | Another approach focusing on actual volatility realized over time.                          |
| II.8.5.8   | Pricing and Hedging Options              | Check model’s option pricing accuracy and hedge performance.                                |
| II.8.6     | Summary and Conclusions                   | Wrap-up of methods to test if your model stands up to real market challenges.               |

