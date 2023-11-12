import numba as nb
import numpy as np

@nb.njit(parallel=True)
def nb_moments(values, moment):
    vals_mean = np.mean(values)
    numerator = (values - vals_mean) ** moment
    numerator = np.mean(numerator )
    return numerator#/len(values)

@nb.njit#(parallel=True)
def nb_return(arr,periods):
    result = np.empty( arr.shape,dtype=np.float64)
    # return
    for i in range(periods,arr.shape[0]):
        if np.isnan(arr[i-periods]):
            result[i] = np.nan
        else:
            result[i] = arr[i]/arr[i-periods]
    return result

@nb.njit(parallel=True)
def nb_rolling_mean(arr,periods):
    result = np.empty( arr.shape,dtype=np.float64)

    for col in nb.prange(arr.shape[0]):
        for i in range(periods,arr.shape[1]+1):
            target = arr[col][i-periods:i].copy()
            result[col][i-1] = np.mean(target[np.isnan(target)])
    return result


@nb.njit(parallel=True)
def nb_rolling_std(arr,periods):
    result = np.empty( arr.shape,dtype=np.float64)

    for col in nb.prange(arr.shape[0]):
        for i in range(periods,arr.shape[1]+1):
            target = arr[col][i-periods:i].copy()
            result[col][i-1] = np.std(target[np.isnan(target)])
    return result

@nb.njit(parallel=True)
def nb_rolling_skew(arr,periods):
    result = np.empty( arr.shape,dtype=np.float64)

    for col in nb.prange(arr.shape[0]):
        for i in range(periods,arr.shape[1]+1):
            target = arr[col][i-periods:i].copy()
            moments = np.std(target)
            if not np.all(np.isnan(target)) and moments != 0:
                m2 = moments**2
                m3 = moments**3
                n = periods

                result[col][i-1] = (np.sqrt(n*(n-1))/ (n-2)) * (m3 / m2 ** 1.5)

    return result
from scipy.stats import kurtosis


@nb.njit(parallel=True)
def nb_rolling_kurt(arr,periods):
    result = np.empty( arr.shape,dtype=np.float64)

    for col in nb.prange(arr.shape[0]):
        for i in range(periods,arr.shape[1]+1):
            target = arr[col][i-periods:i].copy()
            moments = np.std(target)
            if not np.all(np.isnan(target)) and moments != 0:
                m2 = moments**2
                m4 = moments**4
                n = periods

                result[col][i-1] = 1.0/(n-2.)/(n-3.) * ((np.power(n,2)-1.0)*m4/np.power(m2,2) - 3.*np.power(n-1,2))



@nb.njit(parallel=True)
def nb_rolling_qauntile(arr,periods,choose):
    pass

@nb.njit(parallel=True)
def nb_rolling_min(arr,periods):
    result = np.empty(arr.shape,dtype=np.float64)

    for col in nb.prange(arr.shape[0]):
        for i in nb.prange(periods,arr.shape[1]+1):
            target = arr[col][i-periods:i].copy()
            result[col][i-1] = np.nanmin(target)
    return result

@nb.njit(parallel=True)
def nb_rolling_max(arr,periods):
    result = np.empty(arr.shape,dtype=np.float64)

    for col in nb.prange(arr.shape[0]):
        for i in nb.prange(periods,arr.shape[1]+1):
            target = arr[col][i-periods:i].copy()
            result[col][i-1] = np.nanmax(target)
    return result
