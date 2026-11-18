"""Dependency-free recipe shared by the login launcher and GPU worker."""

ARMS = ('continued', 'feature_kd')
PAIRS = ((.30, .30), (.30, .40))
PILOT = dict(seed=42, optimizer_updates=1000, micro_batch_size=2, accumulation=2,
             learning_rate=2e-6, weight_decay=1e-4, clip_grad=5., warmup_updates=50,
             initialization='B1 job 12187 weights only; fresh optimizer in both arms',
             clean_replay=.5, precision='fp32', teacher='frozen B1, 2x Clean canvas, eval/no_grad',
             feature='single fused FPNC tensor', weight=.1,
             loss='channel-RMS-normalized foreground MSE',
             mask='TRAIN GT polygons minus ignored/padding; fractional area at feature resolution',
             checkpoint_selection='final 1000 updates only',
             thresholds='fixed native .30/.30 and historical B1-cal .30/.40; no new grid search',
             inference='teacher and distillation loss omitted, original detector weights/schema')
