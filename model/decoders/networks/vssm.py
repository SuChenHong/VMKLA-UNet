from ...vmamba import VSSLayer_up


def make_vssm_decoder(**kwargs):
    vssm_d = VSSLayer_up(**kwargs)
    return vssm_d