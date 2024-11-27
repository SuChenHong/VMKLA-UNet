from ...vmamba import VSSLayer

def make_vssm_encoder(**kwargs):
    vssm_e = VSSLayer(**kwargs)
    return vssm_e