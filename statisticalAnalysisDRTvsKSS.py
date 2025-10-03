from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the CSV file
file_path = 'C:/Users/caiol/Desktop/PICode/PICode v3/sim_filtered_34.csv'
df = pd.read_csv(file_path)

# Create a column with the user label (e.g., fp01, fp02)
df['User'] = df['Filename'].apply(lambda x: '_'.join(x.split('_')[:1]))

# Create a column for the condition (e.g., _1, _2, etc.)
df['Condition'] = df['Filename'].apply(
    lambda x: x.split('_')[-1].split('.')[0])

# Map conditions to specific colors
condition_palette = {
    '1': 'blue',
    '2': 'orange',
    '3': 'green',
    '4': 'red'
}

# Function to generate plots for a user


def generate_plots(user_data, user_id):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f'Graphs for user: {user_id}', fontsize=16)

    # Graph 1: kss_answer vs pdt_reaction_time with polynomial regression
    X = user_data[['kss_answer']].values
    y = user_data['pdt_reaction_time'].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)

    sns.scatterplot(
        x='kss_answer', y='pdt_reaction_time',
        hue='Condition', palette=condition_palette,
        data=user_data, ax=axes[0, 0]
    )
    sns.lineplot(
        x='kss_answer', y=y_pred,
        data=user_data, color='red', ax=axes[0, 0], label='Polynomial Fit'
    )
    axes[0, 0].set_title('KSS vs DRT with Polynomial Regression')
    axes[0, 0].set_xlabel('KSS')
    axes[0, 0].set_ylabel('DRT')

    # Graph 2: Boxplot of KSS by categories vs DRT
    bins = [0, 3, 5, 7, 9]
    labels = ['Low', 'Moderate', 'High', 'Very High']
    user_data['kss_category'] = pd.cut(
        user_data['kss_answer'], bins=bins, labels=labels)
    sns.boxplot(x='kss_category', y='pdt_reaction_time',
                data=user_data, ax=axes[0, 1])
    axes[0, 1].set_title('KSS Category vs DRT')
    axes[0, 1].set_xlabel('KSS Category')
    axes[0, 1].set_ylabel('DRT')

    # Graph 3: Histogram of pdt_reaction_time
    sns.histplot(user_data['pdt_reaction_time'],
                 kde=True, bins=30, ax=axes[1, 0])
    axes[1, 0].set_title('DRT Histogram')
    axes[1, 0].set_xlabel('DRT')

    # Graph 4: Histogram of kss_answer
    sns.histplot(user_data['kss_answer'], kde=True, bins=30, ax=axes[1, 1])
    axes[1, 1].set_title('KSS Histogram')
    axes[1, 1].set_xlabel('KSS')

    # Graph 5: Polynomial regression of kss_answer over time
    X_time = user_data[['timer [s]']].values
    y_kss = user_data['kss_answer'].values
    poly_time = PolynomialFeatures(degree=2)
    X_time_poly = poly_time.fit_transform(X_time)
    model_kss = LinearRegression().fit(X_time_poly, y_kss)
    y_kss_pred = model_kss.predict(X_time_poly)

    sns.scatterplot(
        x='timer [s]', y='kss_answer',
        hue='Condition', palette=condition_palette,
        data=user_data, ax=axes[2, 0]
    )
    sns.lineplot(
        x='timer [s]', y=y_kss_pred,
        data=user_data, color='red', ax=axes[2, 0], label='Polynomial Fit'
    )
    axes[2, 0].set_title('KSS over Time with Polynomial Regression')
    axes[2, 0].set_xlabel('Time [s]')
    axes[2, 0].set_ylabel('KSS')

    # Graph 6: Polynomial regression of pdt_reaction_time over time
    y_pvt = user_data['pdt_reaction_time'].values
    model_pvt = LinearRegression().fit(X_time_poly, y_pvt)
    y_pvt_pred = model_pvt.predict(X_time_poly)

    sns.scatterplot(
        x='timer [s]', y='pdt_reaction_time',
        hue='Condition', palette=condition_palette,
        data=user_data, ax=axes[2, 1]
    )
    sns.lineplot(
        x='timer [s]', y=y_pvt_pred,
        data=user_data, color='red', ax=axes[2, 1], label='Polynomial Fit'
    )
    axes[2, 1].set_title('DRT over Time with Polynomial Regression')
    axes[2, 1].set_xlabel('Time [s]')
    axes[2, 1].set_ylabel('DRT')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=False)


# Generate plots for each user
for user_id, user_data in df.groupby('User'):
    generate_plots(user_data, user_id)

plt.show(block=True)
