import numpy as np
from scipy import stats

# Generate example data for dependent samples
np.random.seed(0)
pre_test_scores = np.random.normal(loc=10, scale=2, size=100)  # Example pre-test scores
post_test_scores = pre_test_scores + np.random.normal(loc=1, scale=2, size=100)  # Example post-test scores

# Generate example data for independent samples
group1_scores = np.random.normal(loc=10, scale=2, size=100)  # Example scores for group 1
group2_scores = np.random.normal(loc=12, scale=2, size=100)  # Example scores for group 2

# Define hypotheses for dependent samples
null_hypothesis_dep = "There is no difference in the mean scores before and after the intervention"
alternative_hypothesis_dep = "There is a difference in the mean scores before and after the intervention"

# Define hypotheses for independent samples
null_hypothesis_ind = "There is no difference in the mean scores between the two groups"
alternative_hypothesis_ind = "There is a difference in the mean scores between the two groups"

# Perform dependent (paired) t-test
t_statistic_dep, p_value_dep = stats.ttest_rel(pre_test_scores, post_test_scores)

# Perform independent t-test
t_statistic_ind, p_value_ind = stats.ttest_ind(group1_scores, group2_scores)

# Perform z-test (assuming equal variances for simplicity)
z_statistic_ind, p_value_z_ind = stats.ttest_ind(group1_scores, group2_scores)

# Define significance level
alpha = 0.05

# Print results for dependent samples
print("Dependent (Paired) Samples:")
print("Null Hypothesis:", null_hypothesis_dep)
print("Alternative Hypothesis:", alternative_hypothesis_dep)
print("Mean of Pre-Test Scores:", np.mean(pre_test_scores))
print("Mean of Post-Test Scores:", np.mean(post_test_scores))
print("T-statistic (paired):", t_statistic_dep)
print("P-value (paired):", p_value_dep)

# Check for statistical significance for dependent samples
if p_value_dep < alpha:
    print("Result: Reject the null hypothesis for dependent samples")
else:
    print("Result: Fail to reject the null hypothesis for dependent samples")

# Print results for independent samples
print("\nIndependent Samples (t-test):")
print("Null Hypothesis:", null_hypothesis_ind)
print("Alternative Hypothesis:", alternative_hypothesis_ind)
print("Mean of Group 1 Scores:", np.mean(group1_scores))
print("Mean of Group 2 Scores:", np.mean(group2_scores))
print("T-statistic (independent):", t_statistic_ind)
print("P-value (independent):", p_value_ind)

# Check for statistical significance for independent samples
if p_value_ind < alpha:
    print("Result: Reject the null hypothesis for independent samples (t-test)")
else:
    print("Result: Fail to reject the null hypothesis for independent samples (t-test)")

# Print results for independent samples using z-test
print("\nIndependent Samples (z-test):")
print("Null Hypothesis:", null_hypothesis_ind)
print("Alternative Hypothesis:", alternative_hypothesis_ind)
print("Mean of Group 1 Scores:", np.mean(group1_scores))
print("Mean of Group 2 Scores:", np.mean(group2_scores))
print("Z-statistic (independent):", z_statistic_ind)
print("P-value (independent):", p_value_z_ind)

# Check for statistical significance for independent samples using z-test
if p_value_z_ind < alpha:
    print("Result: Reject the null hypothesis for independent samples (z-test)")
else:
    print("Result: Fail to reject the null hypothesis for independent samples (z-test)")
