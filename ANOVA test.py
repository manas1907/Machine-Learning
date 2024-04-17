from scipy.stats import f_oneway

# Sample datasets
data1 = [35, 47, 22, 58, 63]
data2 = [18, 72, 27, 83, 46]
data3 = [56, 63, 77, 38, 99]

# Performing ANOVA test
f_statistic, p_value = f_oneway(data1, data2, data3)

# Printing the results
print("F-statistic:", f_statistic)
print("P-value:", p_value)

# Interpret the p-value
alpha = 0.05  # significance level
if p_value < alpha:
    print("Reject null hypothesis")
else:
    print("Fail to reject null hypothesis, Accepted!!")
