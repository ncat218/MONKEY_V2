"""Exploratory latent pilot recipe; separate from the frozen historical protocol."""
ARMS=('head','plain','gated')
PAIRS=((.30,.40),(.40,.60))
RECIPE=dict(seed=42,optimizer_updates=1000,micro_batch_size=2,accumulation=2,
    learning_rate=2e-5,weight_decay=1e-4,clip_grad=5.,warmup_updates=50,
    scheduler='Linear .1 to 1 for 50 optimizer updates, then Poly .9 to LR*.01',
    initialization='same B1 weights, fresh optimizer; zero residual output',
    frozen='backbone and FPNC parameters and buffers',trainable='DBHead and optional adapter',
    data='same frozen 15222 TRAIN sources; 50/50 Clean/native LR2',
    loss='unchanged DBModuleLoss only; no KD, CRAFT or identity penalty',
    checkpoint_selection='final 1000 updates only',precision='fp32',
    primary_pair=[.40,.60],calibration='none; two fixed operating points',
    plain_hidden=96,gated_hidden=64)
