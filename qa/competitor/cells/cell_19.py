def build_model(unfreeze_last, source=None, variant='small', pool='cls_mean', prior=False):
    from transformers import AutoModel
    p = source if source is not None else DINOV2_SOURCE
    if p is None:
        raise FileNotFoundError('DINOv2 weights not attached')
    bb = AutoModel.from_pretrained(str(p))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size
    trainable = sum((p.numel() for p in bb.parameters() if p.requires_grad))
    log(f'backbone: {n_layer} blocks, last {unfreeze_last} trainable ({trainable / 1000000.0:.1f}M params), feature dim {dim * POOL_PARTS[pool]}')
    return Model(bb, dim, pool=pool, prior=prior)
_speed_original_build_model = build_model
"""Reuse a CPU architecture template; every member still strict-loads and fingerprints."""
import copy as _speed_copy
import threading as _speed_dino_threading

_speed_dino_templates = {}
_speed_dino_lock = _speed_dino_threading.RLock()


def build_model(unfreeze_last, source=None, variant='small', pool='cls_mean', prior=False):
    key = (int(unfreeze_last), None if source is None else str(source), variant,
           pool, bool(prior), N_SLOT, tuple(TARGETS))
    with _speed_dino_lock:
        if key not in _speed_dino_templates:
            template = _speed_original_build_model(unfreeze_last, source, variant, pool, prior)
            assert all(p.device.type == 'cpu' for p in template.parameters())
            _speed_dino_templates[key] = template
        # No mutable state or GPU tensors shared between independently trained models.
        return _speed_copy.deepcopy(_speed_dino_templates[key])