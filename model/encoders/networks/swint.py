from ...SwinTransformer import BasicLayer


def make_swint_encoder(**kwargs):
    swin_e = BasicLayer(**kwargs)
    return swin_e

