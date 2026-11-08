"""CS431 Rev.4 validation-only workflow."""

CONDITIONS = ('clean', 'lr2', 'lr4', 'lr8', 'b_up_lr2')


def require(condition, message):
    if not condition:
        raise ValueError(message)
