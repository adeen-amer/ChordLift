"""Beat-Transformer vendored model + contract spike (Task 3).

Upstream: https://github.com/zhaojw1998/Beat-Transformer (MIT license),
pinned commit 063667fc9e4e11507f9d76dc1154d9db953a85eb.

`DilatedTransformer.py` / `DilatedTransformerLayer.py` are vendored verbatim
(only the intra-package import in DilatedTransformer.py was made relative).
See FINDINGS.md for the verified preprocessing + inference contract.
"""
from .DilatedTransformer import Demixed_DilatedTransformerModel

__all__ = ["Demixed_DilatedTransformerModel"]
