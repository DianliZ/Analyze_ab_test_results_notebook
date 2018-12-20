# Analyze A/B Test Results
You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page. Either way assure that your code passes the project RUBRIC. **Please save regularly

This project will assure you have mastered the subjects covered in the statistics lessons. The hope is to have this project be as comprehensive of these topics as possible. Good luck!

## Table of Contents
Introduction
Part I - Probability
Part II - A/B Test
Part III - Regression

## Introduction
A/B tests are very commonly performed by data analysts and data scientists. It is important that you get some practice working with the difficulties of these

For this project, you will be working to understand the results of an A/B test run by an e-commerce website. Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.

As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question. The labels for each classroom concept are provided for each question. This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria. As a final check, assure you meet all the criteria on the RUBRIC.


## Part I - Probability
To get started, let's import our libraries.

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)
1. Now, read in the ab_data.csv data. Store it in df. Use your dataframe to answer the questions in Quiz 1 of the classroom.

a. Read in the dataset and take a look at the top few rows here:

df=pd.read_csv('ab_data.csv')
df.head()
user_id	timestamp	group	landing_page	converted
0	851104	2017-01-21 22:11:48.556739	control	old_page	0
1	804228	2017-01-12 08:01:45.159739	control	old_page	0
2	661590	2017-01-11 16:55:06.154213	treatment	new_page	0
3	853541	2017-01-08 18:28:03.143765	treatment	new_page	0
4	864975	2017-01-21 01:52:26.210827	control	old_page	1
b. Use the below cell to find the number of rows in the dataset.

df.shape
(294478, 5)
c. The number of unique users in the dataset.

df['user_id'].nunique()
290584
d. The proportion of users converted.

df['user_id'].nunique()/df['user_id'].count()
0.98677660130807732
e. The number of times the new_page and treatment don't line up.

df.query('(group == "treatment" and landing_page != "new_page") or (group != "treatment" and landing_page == "new_page")')['user_id'].count()
3893
f. Do any of the rows have missing values?

df.isnull().values.any()
False
2. For the rows where treatment is not aligned with new_page or control is not aligned with old_page, we cannot be sure if this row truly received the new or old page. Use Quiz 2 in the classroom to provide how we should handle these rows.

a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz. Store your new dataframe in df2.

df2 = df.drop(
    df[( (df.group == 'treatment') & (df.landing_page == 'old_page') ) | \
                ( (df.group == 'control') & (df.landing_page == 'new_page') ) ].index
)
# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
0
3. Use df2 and the cells below to answer questions for Quiz3 in the classroom.

a. How many unique user_ids are in df2?

df2['user_id'].nunique()
290584
b. There is one user_id repeated in df2. What is it?

df2[df2.duplicated(['user_id'], keep=False)]['user_id']
1899    773192
2893    773192
Name: user_id, dtype: int64
c. What is the row information for the repeat user_id?

df2[df2['user_id'] == 773192]
user_id	timestamp	group	landing_page	converted
1899	773192	2017-01-09 05:37:58.781806	treatment	new_page	0
2893	773192	2017-01-14 02:55:59.590927	treatment	new_page	0
d. Remove one of the rows with a duplicate user_id, but keep your dataframe as df2.

df2.drop([1899], inplace=True)
4. Use df2 in the below cells to answer the quiz questions related to Quiz 4 in the classroom.

a. What is the probability of an individual converting regardless of the page they receive?

df2.converted.mean()
0.11959708724499628
b. Given that an individual was in the control group, what is the probability they converted?

df2[df2['group'] == 'control'].converted.mean()
0.1203863045004612
c. Given that an individual was in the treatment group, what is the probability they converted?

df2[df2['group'] == 'treatment'].converted.mean()
0.11880806551510564
d. What is the probability that an individual received the new page?

df2[df2.landing_page == 'new_page'].user_id.count()/df2.shape[0]
0.50006194422266881
e. Use the results in the previous two portions of this question to suggest if you think there is evidence that one page leads to more conversions? Write your response below.

There is no evidence that one page leads to more conversions as the results show that the probability that an individual received the new page is 50% and both converted rates are very close to 12%.


## Part II - A/B Test
Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.

However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time? How long do you run to render a decision that neither page is better than another?

These questions are the difficult parts associated with A/B tests in general.

1. For now, consider you need to make the decision just based on all the data provided. If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be? You can state your hypothesis in terms of words or in terms of  pold  and  pnew , which are the converted rates for the old and new pages.

$H_0:$ $p_{old} \geq p_{new}$ <br><br>
$H_1:$ $p_{old} < p_{new}$

2. Assume under the null hypothesis,  pnew  and  pold  both have "true" success rates equal to the converted success rate regardless of page - that is  pnew and  pold  are equal. Furthermore, assume they are equal to the converted rate in ab_data.csv regardless of the page. 


Use a sample size for each page equal to the ones in ab_data.csv. 


Perform the sampling distribution for the difference in converted between the two pages over 10,000 iterations of calculating an estimate from the null. 


Use the cells below to provide the necessary parts of this simulation. If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem. You can use Quiz 5 in the classroom to make sure you are on the right track.


a. What is the convert rate for  pnew  under the null?

p_new = df2.converted.mean()
print(p_new)
0.119597087245
b. What is the convert rate for  pold  under the null? 


p_old = df2.converted.mean()
print(p_old)
0.119597087245
c. What is  nnew ?

n_new = df2[df2.landing_page == 'new_page'].user_id.count()
print(n_new)
145310
d. What is  nold ?

n_old = df2[df2.landing_page == 'old_page'].user_id.count()
print(n_old)
145274
e. Simulate  nnew  transactions with a convert rate of  pnew  under the null. Store these  nnew  1's and 0's in new_page_converted.

new_page_converted = np.random.choice([0, 1], size=n_new, p=[(1-p_new), p_new])
f. Simulate  nold  transactions with a convert rate of  pold  under the null. Store these  nold  1's and 0's in old_page_converted.

old_page_converted = np.random.choice([0, 1], size=n_old, p=[(1-p_old), p_old])
g. Find  pnew  -  pold  for your simulated values from part (e) and (f).

new_page_converted.mean() - old_page_converted.mean()
0.00056918123125249132
h. Simulate 10,000  pnew  -  pold  values using this same process similarly to the one you calculated in parts a. through g. above. Store all 10,000 values in p_diffs.

new_converted_simulation = np.random.binomial(n_new, p_new, 10000)/n_new
old_converted_simulation = np.random.binomial(n_old, p_old, 10000)/n_old
p_diffs = new_converted_simulation - old_converted_simulation
i. Plot a histogram of the p_diffs. Does this plot look like what you expected? Use the matching problem in the classroom to assure you fully understand what was computed here.

control_conv = df2[df2['group'] == 'control'].converted.mean()

treatment_conv = df2[df2['group'] == 'treatment'].converted.mean()

# expect a normal distribution graph

obs_diff = treatment_conv - control_conv

plt.hist(p_diffs);
plt.axvline(x=obs_diff, color='red');

j. What proportion of the p_diffs are greater than the actual difference observed in ab_data.csv?

p_diffs = np.array(p_diffs)
(p_diffs > obs_diff).mean()
0.90469999999999995
k. In words, explain what you just computed in part j.. What is this value called in scientific studies? What does this value mean in terms of whether or not there is a difference between the new and old pages?

It would be better for Audacity to keep the current page since p-value is 90.48% which indicates that we fail to reject the null hypothesis(Type I error rate of 5%) as the current page has a higher probability of convert rate than the new page

l. We could also use a built-in to achieve similar results. Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let n_old and n_new refer the the number of rows associated with the old page and new pages, respectively.

import statsmodels.api as sm

convert_old = df2.query('group == "control" and converted == 1').shape[0]
convert_new = df2.query('group == "treatment" and converted == 1').shape[0]
n_old = df2[df2['group'] == 'control'].converted.shape[0]
n_new = df2[df2['group'] == 'treatment'].converted.shape[0]
print(convert_old, convert_new, n_old, n_new)
17489 17264 145274 145310
m. Now use stats.proportions_ztest to compute your test statistic and p-value. Here is a helpful link on using the built in.

z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='larger')
z_score, p_value
(1.3109241984234394, 0.094941687240975514)
from scipy.stats import norm

norm.cdf(z_score)
# 0.9999999383005862 # Tells us how significant our z-score is

norm.ppf(1-(0.05/2))
# 1.959963984540054 # Tells us what our critical value at 95% confidence is
1.959963984540054
n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages? Do they agree with the findings in parts j. and k.?

Since the z_score of 1.31092 falls within the critical value of 1.95996, we fail to reject the null hypothesis. This is as the same as the findings in parts j and k.


## Part III - A regression approach
1. In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.


a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

Logistic regression.

b. The goal is to use statsmodels to fit the regression model you specified in part a. to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create a colun for the intercept, and create a dummy variable column for which page each user received. Add an intercept column, as well as an ab_page column, which is 1 when an individual receives the treatment and 0 if control.

df2.head()
user_id	timestamp	group	landing_page	converted
0	851104	2017-01-21 22:11:48.556739	control	old_page	0
1	804228	2017-01-12 08:01:45.159739	control	old_page	0
2	661590	2017-01-11 16:55:06.154213	treatment	new_page	0
3	853541	2017-01-08 18:28:03.143765	treatment	new_page	0
4	864975	2017-01-21 01:52:26.210827	control	old_page	1
df2['intercept'] = 1

df2[['drop', 'ab_page']] = pd.get_dummies(df2['group'])
df2.drop(['drop'], axis=1, inplace=True)
df2.head()
user_id	timestamp	group	landing_page	converted	intercept	ab_page
0	851104	2017-01-21 22:11:48.556739	control	old_page	0	1	0
1	804228	2017-01-12 08:01:45.159739	control	old_page	0	1	0
2	661590	2017-01-11 16:55:06.154213	treatment	new_page	0	1	1
3	853541	2017-01-08 18:28:03.143765	treatment	new_page	0	1	1
4	864975	2017-01-21 01:52:26.210827	control	old_page	1	1	0
c. Use statsmodels to import your regression model. Instantiate the model, and fit the model using the two columns you created in part b. to predict whether or not an individual converts.

import statsmodels.api as sm

logit_mod = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
d. Provide the summary of your model below, and use it as necessary to answer the following questions.

results = logit_mod.fit()
results.summary()
Optimization terminated successfully.
         Current function value: 0.366118
         Iterations 6
Logit Regression Results
Dep. Variable:	converted	No. Observations:	290584
Model:	Logit	Df Residuals:	290582
Method:	MLE	Df Model:	1
Date:	Mon, 10 Dec 2018	Pseudo R-squ.:	8.077e-06
Time:	04:50:02	Log-Likelihood:	-1.0639e+05
converged:	True	LL-Null:	-1.0639e+05
LLR p-value:	0.1899
coef	std err	z	P>|z|	[0.025	0.975]
intercept	-1.9888	0.008	-246.669	0.000	-2.005	-1.973
ab_page	-0.0150	0.011	-1.311	0.190	-0.037	0.007
e. What is the p-value associated with ab_page? Why does it differ from the value you found in the Part II?

Hint: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the Part II?

The p-value associated with ab_page is 0.190 which is different from the p-value of 0.9048 in Part II due to the different hypotheses.

In Part II, unless the new page indicates a higher conversion rate, we will keep the current page. In Part III, a regression test is testing whether the independent variable has any effect.

f. Now, you are considering other things that might influence whether or not an individual converts. Discuss why it is a good idea to consider other factors to add into your regression model. Are there any disadvantages to adding additional terms into your regression model?

Other factors adding into the regression model such as location, gender, date and time would be good ideas to consider as they will have some effects. However, adding more factors will make the model complicated and may not interpret the data clearly.

g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the countries.csv dataset and merge together your datasets on the approporiate rows. Here are the docs for joining tables.

Does it appear that country had an impact on conversion? Don't forget to create dummy variables for these country columns - Hint: You will need two columns for the three dummy varaibles. Provide the statistical output as well as a written response to answer this question.

c_df = pd.read_csv('countries.csv')
C_df2 = c_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')

c_df2.head(2)
user_id	timestamp	group	landing_page	converted	intercept	ab_page	country	CA	UK	US	UK_new_page	US_new_page
0	851104	2017-01-21 22:11:48.556739	control	old_page	0	1	0	US	0	0	1	0	0
1	804228	2017-01-12 08:01:45.159739	control	old_page	0	1	0	US	0	0	1	0	0
# Create dummy variables
c_df2[['CA', 'UK', 'US']] = pd.get_dummies(c_df2['country'])
logit_mod_new = sm.Logit(c_df2['converted'],c_df2[['intercept', 'ab_page', 'US', 'UK']])
results_new = logit_mod_new.fit()
results_new.summary()
Optimization terminated successfully.
         Current function value: 0.366113
         Iterations 6
Logit Regression Results
Dep. Variable:	converted	No. Observations:	290584
Model:	Logit	Df Residuals:	290580
Method:	MLE	Df Model:	3
Date:	Mon, 10 Dec 2018	Pseudo R-squ.:	2.323e-05
Time:	04:50:03	Log-Likelihood:	-1.0639e+05
converged:	True	LL-Null:	-1.0639e+05
LLR p-value:	0.1760
coef	std err	z	P>|z|	[0.025	0.975]
intercept	-2.0300	0.027	-76.249	0.000	-2.082	-1.978
ab_page	-0.0149	0.011	-1.307	0.191	-0.037	0.007
US	0.0408	0.027	1.516	0.130	-0.012	0.093
UK	0.0506	0.028	1.784	0.074	-0.005	0.106
np.exp(-0.0149),np.exp(0.0506),np.exp(0.0408)
(0.9852104557227469, 1.0519020483004984, 1.0416437559600236)
1/np.exp(-0.0149)
1.0150115583846535
The p_values exceed 5% which are not significant. we fail to reject the null hypothesis. The new_page is not significantly better than the current one.

h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion. Create the necessary additional columns, and fit the new model.

Provide the summary results, and your conclusions based on the results.

c_df2['UK_new_page'] = c_df2['ab_page']* c_df2['UK']
c_df2['US_new_page'] = c_df2['ab_page']* c_df2['US']
c_df2.head()
user_id	timestamp	group	landing_page	converted	intercept	ab_page	country	CA	UK	US	UK_new_page	US_new_page
0	851104	2017-01-21 22:11:48.556739	control	old_page	0	1	0	US	0	0	1	0	0
1	804228	2017-01-12 08:01:45.159739	control	old_page	0	1	0	US	0	0	1	0	0
2	661590	2017-01-11 16:55:06.154213	treatment	new_page	0	1	1	US	0	0	1	0	1
3	853541	2017-01-08 18:28:03.143765	treatment	new_page	0	1	1	US	0	0	1	0	1
4	864975	2017-01-21 01:52:26.210827	control	old_page	1	1	0	US	0	0	1	0	0
#Linear Model
lin_mod = sm.OLS(c_df2['converted'], c_df2[['intercept', 'ab_page', 'US', 'US_new_page', 'UK', 'UK_new_page']])
results = lin_mod.fit()
results.summary()
OLS Regression Results
Dep. Variable:	converted	R-squared:	0.000
Model:	OLS	Adj. R-squared:	0.000
Method:	Least Squares	F-statistic:	1.466
Date:	Mon, 10 Dec 2018	Prob (F-statistic):	0.197
Time:	04:50:03	Log-Likelihood:	-85265.
No. Observations:	290584	AIC:	1.705e+05
Df Residuals:	290578	BIC:	1.706e+05
Df Model:	5		
Covariance Type:	nonrobust		
coef	std err	t	P>|t|	[0.025	0.975]
intercept	0.1188	0.004	31.057	0.000	0.111	0.126
ab_page	-0.0069	0.005	-1.277	0.202	-0.017	0.004
US	0.0018	0.004	0.467	0.641	-0.006	0.010
US_new_page	0.0047	0.006	0.845	0.398	-0.006	0.016
UK	0.0012	0.004	0.296	0.767	-0.007	0.009
UK_new_page	0.0080	0.006	1.360	0.174	-0.004	0.020
Omnibus:	125549.436	Durbin-Watson:	1.995
Prob(Omnibus):	0.000	Jarque-Bera (JB):	414285.945
Skew:	2.345	Prob(JB):	0.00
Kurtosis:	6.497	Cond. No.	26.1
np.exp(results.params)
intercept      1.126126
ab_page        0.993143
US             1.001849
US_new_page    1.004727
UK             1.001240
UK_new_page    1.008062
dtype: float64
The p_values are not significant. we fail to reject the null hypothesis. The new_page is not significantly better than the current one.
