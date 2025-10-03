from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
file_path = 'C:/Users/caiol/Desktop/PICode/PICode v3/sim_filtered_34.csv'
df = pd.read_csv(file_path)

# Create a column with the user label
df['User'] = df['Filename'].apply(lambda x: '_'.join(x.split('_')[:1]))

# Init lists to coefficients and correlations
coeff_data = []
correlations_data = []

# Loop to calculate statistics by user
for user, group in df.groupby('User'):
    # Polynomial regression for kss_answer vs pdt_reaction_time
    X_kss = group[['kss_answer']].values
    y_pvt = group['pdt_reaction_time'].values
    poly = PolynomialFeatures(degree=2)
    X_kss_poly = poly.fit_transform(X_kss)
    model_kss_pvt = LinearRegression().fit(X_kss_poly, y_pvt)
    kss_pvt_coeff = model_kss_pvt.coef_[0:]

    # Polynomial regression for kss_answer over time
    X_time = group[['timer [s]']].values
    y_kss = group['kss_answer'].values
    X_time_poly = poly.fit_transform(X_time)
    model_time_kss = LinearRegression().fit(X_time_poly, y_kss)
    time_kss_coeff = model_time_kss.coef_[0:]

    # Polynomial regression for pdt_reaction_time over time
    model_time_pvt = LinearRegression().fit(X_time_poly, y_pvt)
    time_pvt_coeff = model_time_pvt.coef_[0:]

    if group['pdt_reaction_time'].nunique() > 1 and group['kss_answer'].nunique() > 1:
        # Pearson and Spearman correlations
        corr_pearson, p_value_pearson = stats.pearsonr(
            group['pdt_reaction_time'], group['kss_answer'])
        corr_spearman, p_value_spearman = stats.spearmanr(
            group['pdt_reaction_time'], group['kss_answer'])

        # Distance Correlation (non-linear)
        distance_corr_value, distance_corr_pvalue = pg.distance_corr(
            group['kss_answer'], group['pdt_reaction_time']
        )

        # Mutual Information
        mutual_info = mutual_info_regression(
            group[['kss_answer']], group['pdt_reaction_time'])[0]

        print(
            f'\nPearson Correlation for {user}: {corr_pearson:.4f} (p = {p_value_pearson:.4f})')
        print(
            f'Spearman Correlation for {user}: {corr_spearman:.4f} (p = {p_value_spearman:.4f})')
        print(f'Distance Correlation for {user}: {distance_corr_value:.4f}')
        print(f'Mutual Information for {user}: {
              mutual_info:.4f} (p = {distance_corr_pvalue:.4f})')
    else:
        print(f"\nNo correlation for {user}: constant input detected.")

    # Store data
    coeff_data.append([user, *kss_pvt_coeff, *time_kss_coeff, *time_pvt_coeff])
    correlations_data.append(
        [user, corr_pearson, corr_spearman, distance_corr_value, mutual_info])

# Data for DataFrames
coeff_df = pd.DataFrame(coeff_data, columns=['User', 'kss_pvt_0', 'kss_pvt_1st', 'kss_pvt_2nd',
                                             'time_kss_0', 'time_kss_1st', 'time_kss_2nd',
                                             'time_pvt_0', 'time_pvt_1st', 'time_pvt_2nd'])
corr_df = pd.DataFrame(correlations_data, columns=[
                       'User', 'Pearson', 'Spearman', 'Distance Correlation', 'Mutual Information'])

# Hypothesis testing for coefficients (ANOVA)
coeff_columns = coeff_df.columns[1:]
anova_results = {}

for col in coeff_columns:
    anova = pg.anova(dv=col, between='User', data=coeff_df, detailed=True)
    anova_results[col] = anova

# Display ANOVA results
for col, result in anova_results.items():
    print(f"\nANOVA for {col}:")
    print(result)

# Hypothesis testing for correlations (t-tests)
test_results = []

for metric in ['Pearson', 'Spearman', 'Distance Correlation', 'Mutual Information']:
    metric_values = corr_df[metric].dropna()
    if len(metric_values) > 1:
        t_stat, p_value = stats.ttest_1samp(metric_values, 0)
        test_results.append((metric, t_stat, p_value))

# Display T-test results
for metric, t_stat, p_value in test_results:
    print(f"\nT-test for {metric}: t = {t_stat:.4f}, p = {p_value:.4f}")

# PCA analysis
imputer = SimpleImputer(strategy='mean')
pca_data = pd.concat([coeff_df.iloc[:, 1:], corr_df.iloc[:, 1:]], axis=1)
pca_data_imputed = imputer.fit_transform(pca_data)

# Apply PCA and print the transformation matrix
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_data_imputed)

# Display the PCA components matrix
print("\nPCA components matrix:")
print(pca.components_)

# Visualize the importance of each variable in the first principal component
plt.figure(figsize=(10, 5))
plt.bar(range(len(pca.components_[0])), pca.components_[0])
plt.xticks(range(len(pca_data.columns)), pca_data.columns, rotation=90)
plt.title('Contribution of Each Variable to the First Principal Component')
plt.ylabel('Component Weight')
plt.grid(True)
plt.show(block=False)

# Visualize the importance of each variable in the second principal component
plt.figure(figsize=(10, 5))
plt.bar(range(len(pca.components_[1])), pca.components_[1])
plt.xticks(range(len(pca_data.columns)), pca_data.columns, rotation=90)
plt.title('Contribution of Each Variable to the Second Principal Component')
plt.ylabel('Component Weight')
plt.grid(True)
plt.show(block=False)

# Scatter plot of PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', marker='o')
user_ids = df['User'].unique()  # User IDs

# Add user IDs
for i, user_id in enumerate(user_ids):
    plt.text(pca_result[i, 0], pca_result[i, 1], user_id, fontsize=9)

plt.title('PCA of Polynomial Coefficients and Correlations')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show(block=False)

# Clusters based on KSS and DRT coefficients
kmeans_coeff = KMeans(n_clusters=3).fit(coeff_df.iloc[:, 1:])
coeff_df['Cluster'] = kmeans_coeff.labels_

plt.figure(figsize=(8, 6))
sns.scatterplot(x='time_kss_1st', y='time_pvt_1st',
                hue='Cluster', data=coeff_df, palette='viridis')

# Add users
for i, user_id in enumerate(user_ids):
    plt.text(coeff_df['time_kss_1st'].iloc[i],
             coeff_df['time_pvt_1st'].iloc[i], user_id, fontsize=9)

plt.title('Clusters Based on KSS and DRT Coefficients')
plt.xlabel('1st Degree Coefficient (KSS vs time)')
plt.ylabel('1st Degree Coefficient (DRT vs time)')
plt.grid(True)
plt.show(block=False)

# Clusters based on Pearson, Spearman, and Mutual Information
corr_df_imputed = pd.DataFrame(
    imputer.fit_transform(corr_df.iloc[:, 1:]), columns=corr_df.columns[1:])

# Apply KMeans after treating NaNs
kmeans_corr = KMeans(n_clusters=3).fit(corr_df_imputed)
corr_df_imputed['Cluster'] = kmeans_corr.labels_

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Pearson', y='Mutual Information', hue='Cluster',
                data=corr_df_imputed, palette='viridis')

# Add users
for i, user_id in enumerate(user_ids):
    plt.text(corr_df_imputed['Pearson'][i],
             corr_df_imputed['Mutual Information'][i], user_id, fontsize=9)

plt.title('Clusters Based on Pearson and Mutual Information (Imputed)')
plt.xlabel('Pearson Correlation')
plt.ylabel('Mutual Information')
plt.grid(True)
plt.show(block=True)
