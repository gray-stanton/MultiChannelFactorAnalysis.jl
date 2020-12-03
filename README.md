# Multi-Channel Factor Analysis
An implementation of Algorithm 1: ML-MFA from:

>D. Ramírez, I. Santamaria, L. L. Scharf and S. Van Vaerenbergh, "Multi-Channel Factor Analysis With Common and Unique Factors," in IEEE Transactions on Signal Processing, vol. 68, pp. 113-126, 2020, doi: 10.1109/TSP.2019.2955829.

## Entrypoint
Call the *fitMFA* function, which returns the estimated loading matrices for common and unique factors and the estimated diagonal error covariance matrix.

Arguments:
1. *data*: A 3-way ragged array, whose first axis is sample, second axis is channel, and third axis observation 
2. *Ls*: Number of observations per channel
3. *p*: Number of common factors to fit
4. *pjs*: Array whose *i*th entry is the number of unique factors for channel *i*

Optional arguments are *tol* and *maxiter*.



