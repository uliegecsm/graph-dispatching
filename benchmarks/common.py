def asymptotic_speedup(T_E, T_G_sr):
    return T_E / T_G_sr

def n_be(T_G_d, T_G_i, T_E, T_G_s0, T_G_sr):
    return ( (T_G_d + T_G_i) + (T_G_s0 - T_G_sr) )/ (T_E - T_G_sr)

def step_like_plot(ax, x, y, predicate, color):
    assert len(x) == len(y)
    for i in range(len(x)):
        if predicate[i]:
            # Special case: first point.
            if i == 0:
                left = x[i]
                right = (x[i] + x[i+1]) / 2.
            # Special case: last point.
            elif i == len(x) - 1:
                left = (x[i-1] + x[i]) / 2
                right = x[i]
            # General case.
            else:
                left = (x[i-1] + x[i]) / 2
                right = (x[i] + x[i+1]) / 2
            ax.hlines(y[i], left, right, color=color)
    ax.scatter(x[predicate], y[predicate], marker='.', color=color)

def S_n(n_T_E, T_G_d, T_G_i, T_G_s0, n_m1_T_G_sr):
    return n_T_E / (T_G_d + T_G_i + T_G_s0 + n_m1_T_G_sr)
