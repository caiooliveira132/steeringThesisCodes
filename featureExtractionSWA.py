import numpy as np
import pandas as pd
import random
from scipy.stats import skew, kurtosis

# Entropy functions


def trad_steering_entropy(angles, smoothing=1e-6):
    angles = np.asarray(angles)
    n = len(angles)
    if n < 4:
        return np.nan
    errors = [
        angles[t] - (2.5*angles[t-1] - 2.0*angles[t-2] + 0.5*angles[t-3])
        for t in range(3, n)
    ]
    alpha = np.percentile(np.abs(errors), 90)
    bins = np.linspace(-alpha, alpha, 10)
    counts, _ = np.histogram(errors, bins=bins)
    p = (counts + smoothing) / (counts.sum() + smoothing*len(counts))
    return -np.sum(p * np.log(p) / np.log(len(bins)-1))


def approximate_steering_entropy(U, m, r, eps=1e-12):
    U = np.asarray(U)
    N = len(U)
    if N < m+1:
        return np.nan

    def phi(dim):
        P = np.array([U[i:i+dim] for i in range(N-dim+1)])
        Cs = [
            max(np.sum(np.max(np.abs(P - xi), axis=1) <= r)/(N-dim+1), eps)
            for xi in P
        ]
        return np.mean(np.log(Cs))
    return phi(m) - phi(m+1)


def sample_steering_entropy(U, m, r, eps=1e-12):
    U = np.asarray(U)
    N = len(U)
    if N < m+1:
        return np.nan

    def B(dim):
        P = np.array([U[i:i+dim] for i in range(N-dim+1)])
        L = len(P)
        total = sum(np.sum(np.max(np.abs(P - xi), axis=1) <= r) - 1 for xi in P)
        return max(total / (L*(L-1)), eps)
    return -np.log(B(m+1) / B(m))

# Augmentation function


def augment_transform(swa, scale_factors, offset_values):
    scale = random.choice(scale_factors)
    offset = random.choice(offset_values)
    return swa * scale + offset


# Parameters
window_size = 600
step_size = 600
m_approx = 4
m_sample = 6

scale_factors = [0.5, 0.9, 1.0, 1.1]
offset_values = [-14, -7, 7, 14]

# Load and prepare dataset
df = pd.read_csv('filtered_dataset_v2_20Hz.csv')
df = (
    df[['timer [s]', 'steering_wheel_angle', 'kss_answer', 'Filename', 'User']]
    .dropna()
    .sort_values('timer [s]')
    .rename(columns={
        'timer [s]': 'Timer',
        'steering_wheel_angle': 'SWA',
        'kss_answer': 'KSS',
        'Filename': 'situation',
        'User': 'user'
    })
    .reset_index(drop=True)
)
df['KSS_state'] = (df['KSS'] >= 8).astype(int)

# Extract base windows
records = []
for start in range(0, len(df) - window_size + 1, step_size):
    seg = df.iloc[start:start+window_size]
    kss_mean = seg['KSS'].iloc[-1]
    # skip exactly neutral, e.g. KSS==7
    if np.isclose(kss_mean, 7):
        continue
    swa = seg['SWA'].values

    base = {
        'User':      seg['user'].iat[0],
        'Filename':  seg['situation'].iat[0],
        'KSS_state': int(kss_mean >= 8),
        'WindowStart': start,
        'WindowEnd':   start+window_size,
        'KSS_mean':    kss_mean,
        'SWA_array':   swa,
        'TradSE':      trad_steering_entropy(swa),
        'ApproxSE':    approximate_steering_entropy(swa, m_approx, 0.65*swa.std()),
        'SampleSE':    sample_steering_entropy(swa, m_sample, 0.65*swa.std()),
        'Skewness':    skew(swa),
        'Kurtosis':    kurtosis(swa),
        'StdDev':      np.std(swa),
        'Peak2Peak':   np.ptp(swa)
    }
    records.append(base)
win_df = pd.DataFrame(records)

# Compute needed synthetics
n_alert = (win_df['KSS_state'] == 0).sum()
n_drowsy = (win_df['KSS_state'] == 1).sum()
need_syn = n_alert - n_drowsy
print(f"Alert: {n_alert}, Drowsy: {n_drowsy}, Need +{need_syn} synthetics")

# Generate synthetic drowsy
drowsy_df = win_df[win_df['KSS_state'] == 1].reset_index(drop=True)
augmented = []
for _ in range(need_syn):
    row = drowsy_df.sample(1).iloc[0]
    swa_aug = augment_transform(row['SWA_array'], scale_factors, offset_values)
    r = 0.65 * swa_aug.std()
    new = {**row.drop(labels=['SWA_array']),
           'SWA_array': swa_aug,
           'TradSE': trad_steering_entropy(swa_aug),
           'ApproxSE': approximate_steering_entropy(swa_aug, m_approx, r),
           'SampleSE': sample_steering_entropy(swa_aug, m_sample, r),
           'Skewness': skew(swa_aug),
           'Kurtosis': kurtosis(swa_aug),
           'StdDev': np.std(swa_aug),
           'Peak2Peak': np.ptp(swa_aug)
           }
    augmented.append(new)

# Save
final_df = pd.concat([win_df, pd.DataFrame(augmented)], ignore_index=True)
final_df = final_df.drop(columns=['SWA_array'])
final_df.to_csv('newResults_outdoor_drowsy_augmented.csv', index=False)

print("Balanced dataset saved with counts:",
      final_df['KSS_state'].value_counts().to_dict())
