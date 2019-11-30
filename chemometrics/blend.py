partition_size = 0.001
B_hv = hdc.gen_rand_hv(D)
"b": blend
def blend(absorbances, min_abs_hv, min_wn_hv, D, n=1):
    prod_hv = np.ones(D)
    sum_hv = np.zeros(D)

    for i in range(0,len(absorbances)):
        if (absorbances[i] < threshold):
            continue
        lp = round(absorbances[i],3)
        up = lp + partition_size
        noefup = int(D * (absorbances[i] - lp) / partition_size)
        noeflp = D - noefup
        lp_hv = np.roll(B_hv, int(lp * 1000))
        up_hv = np.roll(B_hv, int(up * 1000))
        abs_hv = np.concatenate((lp_hv[0 : noeflp], up_hv[noeflp:]))
        wavenum_hv = calc_wn_iM(min_wn_hv, i, D, m=(len(absorbances) + 1))
        prod_hv = abs_hv * wavenum_hv
        sum_hv += prod_hv
    return sum_hv
