"""Externally reported content-address anchor for the v3 data freeze receipt.

Only this two-constant module changes when the implementation receipt is
created.  The verifier itself is part of the frozen implementation closure.
The final hash of this anchor is reported before any GPU work so that the
approval record, rather than another mutable repository file, is the trust
root.
"""

FREEZE_RECEIPT_FILE_SHA256 = "f1ed58a4363b8ca82599dd6f82360db71b17dcb5471041e3acf9fee01e221e48"
FREEZE_RECEIPT_LOGICAL_SHA256 = (
    "cd057634ab51bbd818e9e4ac92535e64a2bccf56f8505b96e03f6bcb9225e455"
)

__all__ = ["FREEZE_RECEIPT_FILE_SHA256", "FREEZE_RECEIPT_LOGICAL_SHA256"]
