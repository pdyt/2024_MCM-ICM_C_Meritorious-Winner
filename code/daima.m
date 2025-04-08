def logistic(x):
    return 1 / (1 + np.exp(-x))
def sample_inv_gamma(alpha, beta):
    return stats.invgamma.rvs(alpha, scale=beta)
def polya_gamma_sample(b, size=1, truncation=50):
    samples = np.zeros(size)
    for k in range(1, truncation + 1):
        samples += np.random.gamma(shape=b, scale=1.0 / ((k - 0.5) ** 2 * np.pi ** 2), size=size)
    return samples
def gibbs_sampling(X, y, num_samples, burn_in, tau_sq = 1, xi = 1): 
    # MLE
    model = LogisticRegression().fit(X, y)
    beta_mle = model.coef_[0]
    beta_mle0 = model.intercept_[0]
    n, p = X.shape
    beta = beta_mle
    beta_0 = 0
    lambda_sq = np.ones(p)/10
    nu = np.ones(p)
    omega = np.ones(n)
    beta_samples = []
    tau_samples = []
    sigma2_samples = []
    for _ in tqdm(range(num_samples + burn_in)):
        # update omega
        psi = beta_0 + np.dot(X, beta)
        for i in range(n):
            omega[i] = polya_gamma_sample(1, size=1, truncation=80)
        # update beta ºÍ beta_0
        Omega_diag = np.diag(omega)
        A = np.dot(X.T, Omega_diag @ X) + np.diag(1 / (tau_sq * lambda_sq))
        A_inv = np.linalg.inv(A)
        z = y - 0.5 + omega * psi
        beta_mean = A_inv @ X.T @ Omega_diag @ z
        beta = stats.multivariate_normal.rvs(mean=beta_mean, cov=A_inv)
        beta_0 = np.random.normal(np.sum(omega * (z - np.dot(X, beta))) / np.sum(omega), 1 / np.sqrt(np.sum(omega)))
        # update lambda_sq, tau_sq, nu, xi
        for j in range(p):
            lambda_sq[j] = 1*sample_inv_gamma(1/2, 1/nu[j] + beta[j]**2 / (2 * tau_sq))
            nu[j] = sample_inv_gamma(1/2, 1 + 1/lambda_sq[j])
        tau_sq = 1.5*sample_inv_gamma((p+1)/2, 1/xi + np.sum(beta**2 / lambda_sq) / 2)
        xi = sample_inv_gamma(1/2, 1 + 1/tau_sq)
        if _ >= burn_in:
            beta_samples.append(beta.copy())
            tau_samples.append(tau_sq)
            sigma2_samples.append(1 / np.mean(omega))
    beta_samples = np.array(beta_samples)
    BetaHat = np.mean(beta_samples, axis=0)
    BetaMedian = np.median(beta_samples, axis=0)
    LeftCI = np.percentile(beta_samples, 5, axis=0)
    RightCI = np.percentile(beta_samples, 95, axis=0)
    TauHat = np.mean(tau_samples)
    Sigma2Hat = np.mean(sigma2_samples)
    return {
        "BetaHat": BetaHat, "Beta0": beta_0, "LeftCI": LeftCI,"RightCI": RightCI,"BetaMedian": BetaMedian,"Sigma2Hat": Sigma2Hat,"TauHat": TauHat,"BetaSamples": beta_samples,"TauSamples": tau_samples,"Sigma2Samples": sigma2_samples,"BetaMLE": beta_mle,"BetaMLE0":beta_mle0
    }


def update_and_predict(y, alpha=1, beta=1):
    predictions = []
    for i in range(1, len(y) + 1):
        alpha_n = alpha + np.sum(y[:i])
        beta_n = beta + i - np.sum(y[:i])
        theta_pred = alpha_n / (alpha_n + beta_n)
        predictions.append(theta_pred)
    return predictions
predictions = update_and_predict(y)
absolute_accuracy, relative_accuracy = calculate_accuracy_metrics(y, predictions)
