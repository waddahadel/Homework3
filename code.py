import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)

def run_experiment(n, d, B, n_test=5000, n_reps=30):
    mses = []
    norms = []
    for _ in range(n_reps):
        w_star = np.zeros(d)
        w_star[0] = B

        X_train = np.random.uniform(-1/np.sqrt(d), 1/np.sqrt(d), size=(n, d))
        xi_train = np.random.uniform(-0.1, 0.1, size=n)
        y_train = X_train @ w_star + xi_train

        X_test = np.random.uniform(-1/np.sqrt(d), 1/np.sqrt(d), size=(n_test, d))
        xi_test = np.random.uniform(-0.1, 0.1, size=n_test)
        y_test = X_test @ w_star + xi_test

        model = LinearRegression(fit_intercept=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = np.mean((y_pred - y_test)**2)
        norm = np.linalg.norm(model.coef_)
        mses.append(mse)
        norms.append(norm)
    return np.mean(mses), np.mean(norms)

# part (a)/(b): vary dimension
n, B = 400, 2
dims = [5, 20, 100, 500]
mse_by_dim, norm_by_dim = [], []
for d in dims:
    mse, norm = run_experiment(n, d, B)
    mse_by_dim.append(mse)
    norm_by_dim.append(norm)
    print(f"d={d:4d} | MSE={mse:.5f} | norm={norm:.4f}")

# part (c)/(d): vary norm B
d = 100
norms_B = [0.5, 1, 2, 4]
mse_by_B, norm_by_B = [], []
for Bval in norms_B:
    mse, norm = run_experiment(n, d, Bval)
    mse_by_B.append(mse)
    norm_by_B.append(norm)
    print(f"B={Bval:.1f} | MSE={mse:.5f} | norm={norm:.4f}")


# plot 1: MSE vs dimension
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor('#0f1117')
for ax in axes:
    ax.set_facecolor('#161b27')
    ax.tick_params(colors='#9ca3af', labelsize=10)
    ax.xaxis.label.set_color('#e5e7eb')
    ax.yaxis.label.set_color('#e5e7eb')
    ax.title.set_color('#f9fafb')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3748')

axes[0].plot(dims, mse_by_dim, 'o-', color='#60a5fa', linewidth=2, markersize=8, markerfacecolor='#1e40af')
axes[0].set_xlabel('Dimension d', fontsize=12)
axes[0].set_ylabel('Avg Test MSE', fontsize=12)
axes[0].set_title('Test MSE vs Dimension\n(n=400, B=2)', fontsize=12)
axes[0].set_xticks(dims)
axes[0].grid(True, color='#2d3748', linewidth=0.5)

axes[1].plot(dims, norm_by_dim, 's-', color='#34d399', linewidth=2, markersize=8, markerfacecolor='#065f46')
axes[1].set_xlabel('Dimension d', fontsize=12)
axes[1].set_ylabel('Avg Fitted Norm ||ŵ||₂', fontsize=12)
axes[1].set_title('Fitted Norm vs Dimension\n(n=400, B=2)', fontsize=12)
axes[1].set_xticks(dims)
axes[1].grid(True, color='#2d3748', linewidth=0.5)
# adding reference line for true norm
axes[1].axhline(B, color='#f87171', linestyle='--', linewidth=1.5, label=f'True norm B={B}')
axes[1].legend(fontsize=10, facecolor='#161b27', labelcolor='#e5e7eb')

plt.tight_layout(pad=1.5)
plt.savefig('/mnt/user-data/outputs/plot_dimension.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()

#  plot 2: MSE vs B 
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#161b27')
ax.tick_params(colors='#9ca3af', labelsize=10)
ax.xaxis.label.set_color('#e5e7eb')
ax.yaxis.label.set_color('#e5e7eb')
ax.title.set_color('#f9fafb')
for spine in ax.spines.values():
    spine.set_edgecolor('#2d3748')

ax.plot(norms_B, mse_by_B, 'o-', color='#f472b6', linewidth=2, markersize=8, markerfacecolor='#9d174d')
ax.set_xlabel('Target Norm B', fontsize=12)
ax.set_ylabel('Avg Test MSE', fontsize=12)
ax.set_title('Test MSE vs Target Norm B\n(n=400, d=100)', fontsize=12)
ax.set_xticks(norms_B)
ax.grid(True, color='#2d3748', linewidth=0.5)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/plot_norm.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("Done. Plots saved.")