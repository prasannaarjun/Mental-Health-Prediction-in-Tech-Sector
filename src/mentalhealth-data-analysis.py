import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# import file in a variable
file_path = '../dataset/mental health data set.csv'

# Read the CSV file into a DataFrame
mentalhealth_df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(mentalhealth_df.head())

#Data Cleaning
#dealing with missing data
#to get rid of the variables "Timestamp",“comments”, “state” as these attributes are either string values or date values
mentalhealth_df = mentalhealth_df.drop(['comments'], axis= 1)
mentalhealth_df = mentalhealth_df.drop(['state'], axis= 1)
mentalhealth_df = mentalhealth_df.drop(['Timestamp'], axis=1)
print(mentalhealth_df.head())

mentalhealth_df.head(5)
# Assign default values for each data type
defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

# Create lists by data tpe
intFeatures = ['Age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                 'seek_help']
floatFeatures = []

for feature in mentalhealth_df:
    if feature in intFeatures:
        mentalhealth_df[feature] = mentalhealth_df[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        mentalhealth_df[feature] = mentalhealth_df[feature].fillna(defaultString)
    elif feature in floatFeatures:
        mentalhealth_df[feature] = mentalhealth_df[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not recognized.' % feature)

mentalhealth_df.head(5)

gender=mentalhealth_df['Gender'].str.lower()
gender=mentalhealth_df['Gender'].unique()
#print(gender_unique)

male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

#parsing the gender lists reasign gender values

for (row, col) in mentalhealth_df.iterrows():

    if str.lower(col.Gender) in male_str:
        mentalhealth_df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_str:
        mentalhealth_df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans_str:
        mentalhealth_df['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

#removing invalid entries
stk_list = ['A little about you', 'p']
mentalhealth_df = mentalhealth_df[~mentalhealth_df['Gender'].isin(stk_list)]

#printing gender reassigned gender
print(mentalhealth_df['Gender'].unique())

##Fixing missing values in Age column
# Replace missing age values with the median
mentalhealth_df['Age'] = mentalhealth_df['Age'].fillna(mentalhealth_df['Age'].median())

# Ensure we are modifying the original DataFrame properly
mentalhealth_df.loc[mentalhealth_df['Age'] < 18, 'Age'] = mentalhealth_df['Age'].median()
mentalhealth_df.loc[mentalhealth_df['Age'] > 120, 'Age'] = mentalhealth_df['Age'].median()

# Define Age Ranges
mentalhealth_df['age_range'] = pd.cut(
    mentalhealth_df['Age'], bins=[0, 20, 30, 65, 100],
    labels=["0-20", "21-30", "31-65", "66-100"],
    include_lowest=True
)
print(mentalhealth_df['age_range'].head())


column_headers = mentalhealth_df.columns.tolist()
print(column_headers)

mentalhealth_df['self_employed'] = mentalhealth_df['self_employed'].replace([defaultString], 'No')
print(mentalhealth_df['self_employed'].unique())

mentalhealth_df['work_interfere'] = mentalhealth_df['work_interfere'].replace([defaultString], 'Don\'t know' )
print(mentalhealth_df['work_interfere'].unique())

labelDictionary = {}
for feature in mentalhealth_df:
    le = preprocessing.LabelEncoder()
    le.fit(mentalhealth_df[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    mentalhealth_df[feature] = le.transform(mentalhealth_df[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDictionary[labelKey] = labelValue

for key, value in labelDictionary.items():
    print(key, value)

# Remove 'Country' attribute
mentalhealth_df = mentalhealth_df.drop(['Country'], axis=1)
mentalhealth_df.head()
print(mentalhealth_df)

# Compute the correlation matrix
corrmat = mentalhealth_df.corr()

plt.figure(figsize=(16, 12))  # Bigger figure for better visibility
sns.set(font_scale=1.4)  # Bigger font size
sns.heatmap(
    corrmat, vmin=-1, vmax=1, center=0, square=True, annot=True, fmt=".1f",
    cmap="coolwarm", linewidths=1.5, cbar_kws={'shrink': 0.75}
)
plt.title("Correlation Matrix Heatmap", fontsize=18, fontweight="bold")
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate and increase font
plt.yticks(rotation=0, fontsize=12)  # Ensure y-labels are clear
plt.show()

k = 10  # Number of most correlated features
cols = corrmat.nlargest(k, 'treatment')['treatment'].index
cm = np.corrcoef(mentalhealth_df[cols].values.T)

plt.figure(figsize=(14, 10))  # Wider figure to avoid label cutoff
sns.set(font_scale=1.3)
hm = sns.heatmap(
    cm, cbar=True, annot=True, square=True, fmt='.2f',
    annot_kws={'size': 14}, yticklabels=cols.values, xticklabels=cols.values,
    cmap="coolwarm", linewidths=1, cbar_kws={'shrink': 0.8}
)
plt.title("Treatment Correlation Heatmap", fontsize=18, fontweight="bold")
plt.xticks(rotation=35, ha='right', fontsize=14)  # Rotate x-axis labels for visibility
plt.yticks(fontsize=14)
plt.show()


# Set the figure size correctly
plt.figure(figsize=(12, 8))  # Corrected `figsize` argument

# Use sns.histplot instead of sns.distplot (since distplot is deprecated)
sns.histplot(mentalhealth_df["Age"], bins=24, kde=True)

# Add titles and labels
plt.title("Distribution and Density by Age")
plt.xlabel("Age")
plt.ylabel("Density")

# Show the plot
plt.show()

# Separate by treatment or not
g = sns.FacetGrid(mentalhealth_df, col='treatment', height=5)

# Use sns.histplot() instead of sns.distplot()
g.map(sns.histplot, "Age", bins=20, kde=True)

plt.show()
label_age = labelDictionary['label_age_range']

# Use errorbar=None instead of ci=None
g = sns.catplot(
    x="age_range", y="treatment", hue="Gender",
    data=mentalhealth_df, kind="bar",
    errorbar=None, height=5, aspect=2, legend_out=True
)

# Set custom tick labels
g.set_xticklabels(label_age)

# Set plot title and labels
plt.title('Probability of Mental Health Condition')
plt.ylabel('Probability x 100')
plt.xlabel('Age')

# Replace legend labels
new_labels = labelDictionary['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Adjust legend position
g.fig.subplots_adjust(top=0.9, right=0.8)

plt.show()

o = labelDictionary['label_family_history']

# Use errorbar=None instead of ci=None
g = sns.catplot(
    x="family_history", y="treatment", hue="Gender",
    data=mentalhealth_df, kind="bar",
    errorbar=None, height=5, aspect=2, legend_out=True
)

# Set custom tick labels
g.set_xticklabels(o)

# Set plot title and labels
plt.title('Probability of Mental Health Condition')
plt.ylabel('Probability x 100')
plt.xlabel('Family History')

# Replace legend labels
new_labels = labelDictionary['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Adjust legend position
g.fig.subplots_adjust(top=0.9, right=0.8)

plt.show()
o = labelDictionary['label_care_options']

# Use errorbar=None instead of ci=None
g = sns.catplot(
    x="care_options", y="treatment", hue="Gender",
    data=mentalhealth_df, kind="bar",
    errorbar=None, height=5, aspect=2, legend_out=True
)

# Set custom tick labels
g.set_xticklabels(o)

# Set plot title and labels
plt.title('Probability of Mental Health Condition')
plt.ylabel('Probability x 100')
plt.xlabel('Care Options')

# Replace legend labels
new_labels = labelDictionary['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Adjust legend position
g.fig.subplots_adjust(top=0.9, right=0.8)

plt.show()

o = labelDictionary['label_benefits']

# Use errorbar=None instead of ci=None
g = sns.catplot(
    x="benefits", y="treatment", hue="Gender",  # Ensure x="benefits" is correct
    data=mentalhealth_df, kind="bar",
    errorbar=None, height=5, aspect=2, legend_out=True
)

# Set custom tick labels
g.set_xticklabels(o)

# Set plot title and labels
plt.title('Probability of Mental Health Condition')
plt.ylabel('Probability x 100')
plt.xlabel('Benefits')

# Replace legend labels
new_labels = labelDictionary['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Adjust legend position
g.fig.subplots_adjust(top=0.9, right=0.8)

plt.show()

o = labelDictionary['label_work_interfere']

# Use errorbar=None instead of ci=None
g = sns.catplot(
    x="work_interfere", y="treatment", hue="Gender",
    data=mentalhealth_df, kind="bar",
    errorbar=None, height=5, aspect=2, legend_out=True
)

# Set custom tick labels
g.set_xticklabels(o)

# Set plot title and labels
plt.title('Probability of Mental Health Condition')
plt.ylabel('Probability x 100')
plt.xlabel('Work Interfere')

# Replace legend labels
new_labels = labelDictionary['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Adjust legend position
g.fig.subplots_adjust(top=0.9, right=0.8)

plt.show()
#Features Scaling We're going to scale age, because is extremely different from the other ones.
# Scaling Age
scaler = MinMaxScaler()
mentalhealth_df['Age'] = scaler.fit_transform(mentalhealth_df[['Age']])
mentalhealth_df.head()

mentalhealth_df.to_csv('dataset\mentalhealth-data.csv', index=False)