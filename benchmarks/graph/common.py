import enum

class Method(enum.StrEnum):
    EXEC = 'exec'
    GRAPH = 'graph'

def asymptotic_speedup(T_E, T_G_sr):
    return T_E / T_G_sr

def n_be(T_G_d, T_G_i, T_E, T_G_s0, T_G_sr):
    return ( (T_G_d + T_G_i) + (T_G_s0 - T_G_sr) )/ (T_E - T_G_sr)
