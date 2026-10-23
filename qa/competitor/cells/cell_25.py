N_SLOT_TYPES, MASK_IDX = (6, 0)

def segment_softmax(scores, sidx, B):
    T, K = scores.shape
    idx = sidx.unsqueeze(1).expand(-1, K)
    m = torch.full((B, K), float('-inf'), device=scores.device, dtype=scores.dtype)
    m = m.scatter_reduce(0, idx, scores, reduce='amax', include_self=True)
    e = (scores - m[sidx]).exp()
    s = torch.zeros(B, K, device=scores.device, dtype=scores.dtype).index_add_(0, sidx, e)
    return e / s[sidx].clamp(min=1e-06)

class MeanMaxPool(nn.Module):

    def forward(self, f, sidx, B, slot=None, return_attn=False):
        D = f.shape[1]
        cnt = torch.zeros(B, device=f.device, dtype=f.dtype).index_add_(0, sidx, torch.ones(f.shape[0], device=f.device, dtype=f.dtype))
        mean = torch.zeros(B, D, device=f.device, dtype=f.dtype).index_add_(0, sidx, f)
        mean = mean / cnt.clamp(min=1).unsqueeze(1)
        mx = torch.full((B, D), -10000.0, device=f.device, dtype=f.dtype)
        mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), f, reduce='amax', include_self=True)
        return (torch.cat([mean, mx], 1), None)

class LabelAttentionPool(nn.Module):

    def __init__(self, d, n_labels=12, n_heads=4, slot_bias=True):
        super().__init__()
        self.d, self.k, self.h = (d, n_labels, n_heads)
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.key, self.val = (nn.Linear(d, d), nn.Linear(d, d))
        self.slot_bias = nn.Parameter(torch.zeros(n_labels, N_SLOT_TYPES + 1)) if slot_bias else None

    def forward(self, f, sidx, B, slot=None, return_attn=False):
        scores = self.key(f) @ self.q.t() / self.d ** 0.5
        if self.slot_bias is not None and slot is not None:
            scores = scores + self.slot_bias.t()[slot]
        a = segment_softmax(scores, sidx, B)
        out = torch.zeros(B, self.k, self.d, device=f.device, dtype=f.dtype)
        out = out.index_add_(0, sidx, a.unsqueeze(-1) * self.val(f).unsqueeze(1))
        return (out, a)

class TokenXAttnPool(nn.Module):

    def __init__(self, d, n_labels=12, n_heads=6, dropout=0.2):
        super().__init__()
        self.d, self.k = (d, n_labels)
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.slot_emb = nn.Embedding(N_SLOT_TYPES + 1, d, padding_idx=0)
        self.kv_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

    def forward(self, tok, sidx, B, slot=None, return_attn=False):
        T, N, D = tok.shape
        cnt = torch.bincount(sidx, minlength=B)
        S = int(cnt.max().item())
        starts = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(T, device=tok.device) - starts[sidx]
        kv = tok + self.slot_emb(slot).unsqueeze(1)
        pad = tok.new_zeros(B, S, N, D)
        pad[sidx, pos] = kv
        keep = torch.zeros(B, S, dtype=torch.bool, device=tok.device)
        keep[sidx, pos] = True
        kpm = ~keep.repeat_interleave(N, dim=1)
        pad = self.kv_norm(pad.reshape(B, S * N, D))
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        att, w = self.attn(q, pad, pad, key_padding_mask=kpm, need_weights=return_attn, average_attn_weights=True)
        cls = tok[:, 0]
        mean = torch.zeros(B, D, device=tok.device, dtype=tok.dtype).index_add_(0, sidx, cls) / cnt.clamp(min=1).unsqueeze(1)
        mx = torch.full((B, D), -10000.0, device=tok.device, dtype=tok.dtype)
        mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), cls, reduce='amax', include_self=True)
        base = torch.cat([mean, mx], 1).unsqueeze(1).expand(-1, self.k, -1)
        return (torch.cat([att, base], -1), w)

class ViTSlotToken(nn.Module):

    def __init__(self, vit, n_cat, dim=None):
        super().__init__()
        self.vit = vit
        d = dim or vit.embed_dim
        self.tok = nn.Embedding(n_cat + 1, d, padding_idx=MASK_IDX)
        self.num_features = vit.num_features
        self._orig_prefix = getattr(vit, 'num_prefix_tokens', 1)
        vit.num_prefix_tokens = self._orig_prefix + 1
        for blk in vit.blocks:
            a = getattr(blk, 'attn', None)
            if a is not None and hasattr(a, 'num_prefix_tokens'):
                a.num_prefix_tokens = a.num_prefix_tokens + 1

    @staticmethod
    def _maybe(mod, x):
        return x if mod is None else mod(x)

    def forward_features(self, x, cat):
        v = self.vit
        x = v.patch_embed(x)
        pos = v._pos_embed(x)
        rope = None
        if isinstance(pos, tuple):
            x, rope = pos
        else:
            x = pos
        x = self._maybe(getattr(v, 'patch_drop', None), x)
        x = self._maybe(getattr(v, 'norm_pre', None), x)
        npt = self._orig_prefix
        tok = self.tok(cat).unsqueeze(1)
        x = torch.cat([x[:, :npt], tok, x[:, npt:]], dim=1)
        if rope is not None:
            if getattr(v, 'rope_mixed', False):
                for i, blk in enumerate(v.blocks):
                    x = blk(x, rope=rope[i])
            else:
                for blk in v.blocks:
                    x = blk(x, rope=rope)
        else:
            x = v.blocks(x)
        return v.norm(x)

    def forward_head(self, x, pre_logits=True):
        return self.vit.forward_head(x, pre_logits=pre_logits)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class _GatedDepthBlock(nn.Module):

    def __init__(self, n_slice, dropout=0.0, ls_init=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(1, n_slice)
        self.v = nn.Conv2d(n_slice, n_slice, 1)
        self.g = nn.Conv2d(n_slice, n_slice, 1)
        self.out = nn.Conv2d(n_slice, n_slice, 1)
        self.gamma = nn.Parameter(torch.full((n_slice, 1, 1), ls_init))
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, x):
        z = self.norm(x)
        return x + self.gamma * self.drop(self.out(self.v(z) * F.silu(self.g(z))))

class DepthCompress(nn.Module):

    def __init__(self, n_slice=16, out_ch=3, depth=1, dropout=0.0, ls_init=0.1, imagenet=True, proj_noise=0.25):
        super().__init__()
        self.imagenet = imagenet
        self.blocks = nn.ModuleList([_GatedDepthBlock(n_slice, dropout, ls_init) for _ in range(depth)])
        self.proj = nn.Conv2d(n_slice, out_ch, 1, bias=True)
        if imagenet:
            self.register_buffer('mu', torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1))
            self.register_buffer('sd', torch.tensor(IMAGENET_STD).view(1, -1, 1, 1))

    def forward(self, x):
        keep = (x.amax(dim=1, keepdim=True) > 0).to(x.dtype)
        z = x
        for b in self.blocks:
            z = b(z)
        z = self.proj(z)
        if self.imagenet:
            z = (z - self.mu.to(z.dtype)) / self.sd.to(z.dtype)
        return z * keep
N_PLANE, N_CONTRAST = (3, 2)
_PLANE_OF = lambda s: torch.clamp(s - 1, 0, 5) // 2
_CONTRAST_OF = lambda s: torch.clamp(s - 1, 0, 5) % 2

class SlotDepthMixer(nn.Module):

    def __init__(self, n_slice=16, ksize=5, alpha_max=0.25):
        super().__init__()
        self.n_slice, self.ksize, self.r = (n_slice, ksize, ksize // 2)
        self.alpha_max = alpha_max
        b = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        self.register_buffer('base', b.log()[self.r:])
        n_u = self.r + 1
        self.shared = nn.Parameter(torch.zeros(n_u))
        self.plane_k = nn.Parameter(torch.zeros(N_PLANE, n_u))
        self.contrast_k = nn.Parameter(torch.zeros(N_CONTRAST, n_u))
        self.g0 = nn.Parameter(torch.zeros(()))
        self.gate_p = nn.Parameter(torch.zeros(N_PLANE))
        self.gate_c = nn.Parameter(torch.zeros(N_CONTRAST))
        idx = torch.arange(n_slice)
        self.register_buffer('off', idx[None, :] - idx[:, None])

    def kernel(self, slot):
        p, c = (_PLANE_OF(slot), _CONTRAST_OF(slot))
        half = self.base + self.shared + self.plane_k[p] + self.contrast_k[c]
        full = torch.cat([half.flip(-1)[..., :self.r], half], dim=-1)
        return F.softmax(full, dim=-1)

    def alpha(self, slot):
        p, c = (_PLANE_OF(slot), _CONTRAST_OF(slot))
        return self.alpha_max * torch.tanh(self.g0 + self.gate_p[p] + self.gate_c[c])

    def forward(self, x, slot, vmask):
        T, S, H, W = x.shape
        if vmask is None:
            raise ValueError('stem=mixer requires the padding mask')
        k = self.kernel(slot)
        v = vmask.to(k.dtype)
        d = self.off + self.r
        inb = (d >= 0) & (d < self.ksize)
        kk = k[:, d.clamp(0, self.ksize - 1)] * inb
        M = kk * v[:, None, :]
        den = M.sum(-1, keepdim=True)
        eye = torch.eye(S, device=x.device, dtype=M.dtype).expand(T, S, S)
        ok = (den > 1e-06) & v[:, :, None].bool()
        M = torch.where(ok, M / den.clamp(min=1e-06), eye)
        a = self.alpha(slot)[:, None, None]
        Aop = ((1.0 - a) * eye + a * M).to(x.dtype)
        if x.is_contiguous(memory_format=torch.channels_last) and (not x.is_contiguous()):
            y = torch.bmm(x.permute(0, 2, 3, 1).reshape(T, H * W, S), Aop.transpose(1, 2))
            return y.reshape(T, H, W, S).permute(0, 3, 1, 2)
        return torch.bmm(Aop, x.reshape(T, S, H * W)).reshape(T, S, H, W)

def _seg_mean_max(v, sidx, B):
    D = v.shape[1]
    cnt = torch.zeros(B, device=v.device, dtype=v.dtype).index_add_(0, sidx, torch.ones(v.shape[0], device=v.device, dtype=v.dtype))
    mean = torch.zeros(B, D, device=v.device, dtype=v.dtype).index_add_(0, sidx, v)
    mean = mean / cnt.clamp(min=1).unsqueeze(1)
    mx = torch.full((B, D), -10000.0, device=v.device, dtype=v.dtype)
    mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), v, reduce='amax', include_self=True)
    return torch.cat([mean, mx], 1)

def _pad_kv(x, sidx, B, norm):
    T, P, D = x.shape
    cnt = torch.bincount(sidx, minlength=B)
    S = int(cnt.max().item())
    starts = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(T, device=x.device) - starts[sidx]
    pad = x.new_zeros(B, S, P, D)
    pad[sidx, pos] = x
    keep = torch.zeros(B, S, dtype=torch.bool, device=x.device)
    keep[sidx, pos] = True
    return (norm(pad.reshape(B, S * P, D)), ~keep.repeat_interleave(P, dim=1))

class _GatedDelta(nn.Module):

    def __init__(self, d, n_labels, n_heads, dropout):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.kv_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.d_norm = nn.LayerNorm(d)
        self.dw = nn.Parameter(torch.randn(n_labels, d) * (1.0 / d ** 0.5))
        self.db = nn.Parameter(torch.zeros(n_labels))
        self.gate = nn.Parameter(torch.zeros(n_labels))

    def delta(self, pat, sidx, B, return_attn):
        kv, kpm = _pad_kv(pat, sidx, B, self.kv_norm)
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        att, w = self.attn(q, kv, kv, key_padding_mask=kpm, need_weights=return_attn, average_attn_weights=True)
        return ((self.d_norm(att) * self.dw).sum(-1) + self.db, w)

class TokenResidualPool(_GatedDelta):

    def __init__(self, d, n_labels=12, n_heads=6, pe=64, dropout=0.2):
        super().__init__(d, n_labels, n_heads, dropout)
        self.base = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(dropout), nn.Linear(2 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        base = self.base(torch.cat([_seg_mean_max(tok[:, 1:].mean(1), sidx, B), pres], 1))
        d_, w = self.delta(tok[:, 1:], sidx, B, return_attn)
        return (base + self.gate * d_, w)

class CodexResidualPool(_GatedDelta):

    def __init__(self, d, n_labels=12, n_heads=6, pe=64, dropout=0.2):
        super().__init__(d, n_labels, n_heads, dropout)
        self.base = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(dropout), nn.Linear(2 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        base = self.base(torch.cat([_seg_mean_max(tok[:, 0], sidx, B), pres], 1))
        d_, w = self.delta(tok[:, 1:], sidx, B, return_attn)
        return (base + self.gate * d_, w)

class ClsAddPool(nn.Module):

    def __init__(self, d, n_labels=12, pe=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(4 * d + pe), nn.Dropout(dropout), nn.Linear(4 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        return (self.net(torch.cat([_seg_mean_max(tok[:, 1:].mean(1), sidx, B), _seg_mean_max(tok[:, 0], sidx, B), pres], 1)), None)

class Readout(nn.Module):

    def __init__(self, pool, d, n_labels=12, pe=64):
        super().__init__()
        self.pool_kind, self.k = (pool, n_labels)
        self.pres_emb = nn.Embedding(N_SLOT_TYPES + 1, pe, padding_idx=0)
        if pool in ('xres', 'clsadd', 'xcodex'):
            self.pool = {'xres': TokenResidualPool, 'clsadd': ClsAddPool, 'xcodex': CodexResidualPool}[pool](d, n_labels, pe=pe)
        elif pool in ('attn', 'xattn'):
            if pool == 'xattn':
                self.pool = TokenXAttnPool(d, n_labels)
                wd = 3 * d + pe
            else:
                self.pool = LabelAttentionPool(d, n_labels)
                wd = d + pe
            self.norm = nn.LayerNorm(wd)
            self.w = nn.Parameter(torch.randn(n_labels, wd) * (1.0 / wd ** 0.5))
            self.b = nn.Parameter(torch.zeros(n_labels))
        else:
            self.pool = MeanMaxPool()
            self.net = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(0.2), nn.Linear(2 * d + pe, n_labels))
        self.drop = nn.Dropout(0.2)

    def forward(self, f, slot, sidx, B, return_attn=False):
        pe = self.pres_emb(slot)
        pres = torch.zeros(B, pe.shape[1], device=f.device, dtype=f.dtype).index_add_(0, sidx, pe)
        if self.pool_kind in ('xres', 'clsadd', 'xcodex'):
            return self.pool(f, slot, sidx, B, pres)[0]
        pooled, attn = self.pool(f, sidx, B, slot=slot, return_attn=return_attn)
        if self.pool_kind in ('attn', 'xattn'):
            x = torch.cat([pooled, pres.unsqueeze(1).expand(-1, self.k, -1)], -1)
            x = self.drop(self.norm(x))
            return (x * self.w).sum(-1) + self.b
        return self.net(torch.cat([pooled, pres], 1))

class Net(nn.Module):

    def __init__(self, enc, cond, n_meta=0, pool='mean_max', stem='native', n_slice=16):
        super().__init__()
        self.enc, self.cond = (enc, cond)
        self.compress = DepthCompress(n_slice, 3) if stem == 'compress' else None
        self.mixer = SlotDepthMixer(n_slice) if stem == 'mixer' else None
        self.tokens = pool in ('xattn', 'xres', 'clsadd', 'xcodex')
        D = enc.num_features
        self.meta_mlp = nn.Sequential(nn.LayerNorm(n_meta), nn.Linear(n_meta, 128), nn.GELU(), nn.Linear(128, D)) if n_meta > 0 else None
        self.readout = Readout(pool, D)
        if cond == 'post':
            self.slot_emb = nn.Embedding(N_SLOT_TYPES + 1, D, padding_idx=MASK_IDX)

    def forward(self, im, slot, smeta, sidx, B, vm=None):
        if self.mixer is not None:
            im = self.mixer(im, slot, vm)
        if self.compress is not None:
            im = self.compress(im)
        f = self.enc.forward_features(im, slot) if self.cond == 'token' else self.enc.forward_features(im)
        if self.tokens:
            inner = getattr(self.enc, 'vit', self.enc)
            orig = getattr(self.enc, '_orig_prefix', getattr(inner, 'num_prefix_tokens', 1))
            f = torch.cat([f[:, :1], f[:, orig:]], 1)
        else:
            f = self.enc.forward_head(f, pre_logits=True)
            if f.dim() > 2:
                f = f.flatten(1)
        ex = (lambda v: v.unsqueeze(1)) if self.tokens else lambda v: v
        if self.cond == 'post':
            f = f + ex(self.slot_emb(slot))
        if self.meta_mlp is not None and smeta.shape[1] > 0:
            mt = self.meta_mlp(smeta)
            f = torch.cat([f, mt.unsqueeze(1)], 1) if self.tokens else f + mt
        return self.readout(f, slot, sidx, B)
models=[]
_A5_LOAD_RECEIPT=[]
_a5_paths=[CKPT/f'm_f{i}.pt' for i in range(5)]
if not all(p.is_file() for p in _a5_paths):
    raise RuntimeError(f'A5 requires m_f0.pt..m_f4.pt; missing {[str(p) for p in _a5_paths if not p.is_file()]}')
_a5_config=None
for ckpt_path in _a5_paths:
    z=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    cfg=z['cfg'];fold=int(z['fold'])
    if ckpt_path.name!=f'm_f{fold}.pt': raise RuntimeError('A5 fold/file identity mismatch')
    fields={'backbone':cfg['backbone'],'cond':cfg['cond'],'pool':cfg['pool'],'img':int(cfg['img']),'n_slice':int(cfg.get('n_slice',16)),'n_meta':int(cfg.get('n_meta',0)),'meta':cfg.get('meta','none'),'stem':cfg.get('stem','native'),'norm':cfg.get('norm','none')}
    if _a5_config is None:_a5_config=fields
    if fields!=_a5_config:raise RuntimeError(f'A5 per-fold input/model contract differs: {ckpt_path}')
    if fields['img']!=SIZE or fields['n_slice']!=N_SLICE or fields['n_meta']!=0:
        raise RuntimeError(f'A5 checkpoint input contract incompatible with loader: {fields}')
    if 'labels' in z and list(z['labels'])!=list(LABELS):raise RuntimeError('A5 label order differs')
    _stem=fields['stem'];_in=3 if _stem=='compress' else fields['n_slice']
    enc=timm.create_model(cfg['backbone'],pretrained=False,num_classes=0,in_chans=_in,**({'img_size':cfg['img']} if 'vit_' in cfg['backbone'] else {}))
    if cfg['cond']=='token':enc=ViTSlotToken(enc,N_SLOT_TYPES)
    m=Net(enc,cfg['cond'],cfg.get('n_meta',0),cfg['pool'],stem=_stem,n_slice=cfg.get('n_slice',16))
    rsna_strict_load(m,z['state_dict'],f'A5 fold {fold}')
    models.append(m.eval());_A5_LOAD_RECEIPT.append({'fold':fold,'file':str(ckpt_path),'sha256':rsna_sha(ckpt_path),'tensor_keys':len(z['state_dict']),'strict':True,'config':fields})
    print(f'A5 strict-loaded {ckpt_path.name}; all encoder and head tensors present')
CFG=dict(cfg)
rsna_json('/kaggle/working/diagnostics/a5_loads.json',_A5_LOAD_RECEIPT)