import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm

plt.rcParams["font.family"] = "Times New Roman"

mean = np.array([0 for _ in range(64)])
cov = np.diag([1 for _ in range(64)])

x = np.random.multivariate_normal(mean, cov, 1000)


x_chi = np.arange(-100, 100, 0.001)
chi_pdf = chi2.pdf(x_chi, df = 64)
dot_prod = np.matmul(x, x.transpose(1, 0))


bool_identity = np.identity(dot_prod.shape[0], dtype=bool)
bool_off_diagonal = np.invert(bool_identity)


diag_dot_prod = np.diag(dot_prod)
off_dot_prod = dot_prod[bool_off_diagonal]


sns.distplot(diag_dot_prod, label = 'Common neighbor pairs')
plt.plot(x_chi, chi_pdf, label = 'Chi-distribution of degree 64', linestyle = '--')

sns.distplot(off_dot_prod, label = 'Non-common neighbor pairs')
# plt.plot(norm_pdf, label = 'Normal-distribution', linestyle = '--')


plt.legend()
# plt.show()
plt.savefig('./diff_dot_prod.png', transparent = True, dpi = 100, bbox_inches = 'tight')
# plt.close()


