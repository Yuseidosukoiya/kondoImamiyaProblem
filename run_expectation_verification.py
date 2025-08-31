import numpy as np

runs = np.array([9.032, 9.021, 9.032, 9.007, 8.943, 8.957])
mean = runs.mean()
std = runs.std(ddof=1)
se = std / np.sqrt(len(runs))
ci95 = (mean - 1.96 * se, mean + 1.96 * se)

print(f"平均: {mean:.3f}")
print(f"標準偏差: {std:.3f}")
print(f"標準誤差: {se:.3f}")
print(f"95%信頼区間: {ci95[0]:.3f} ~ {ci95[1]:.3f}")
