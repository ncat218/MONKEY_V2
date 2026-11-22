"""Exploratory fusion/upsampling study, with a continuous in-job pilot gate."""
ARMS=('nearest','asf','bilinear','dysample')
RECIPE=dict(seed=42,optimizer_updates=10000,pilot_updates=1000,micro_batch_size=2,accumulation=2,
            learning_rate=2e-5,weight_decay=1e-4,clip_grad=5.,warmup_updates=100,
            initialization='B1 shared tensors, fresh optimizer; zero DySample offsets',
            trainable='FPNC and DBHead in every arm',frozen='backbone parameters and buffers only',
            data='locked 15222 TRAIN sources, 50/50 Clean/native LR2',loss='unchanged DBModuleLoss',
            precision='fp32',checkpoint_every_updates=2500,selection='final 10000 only; no tuning',
            pilot='first 1000 updates of same 10000-update schedule, technical gate only; no restart',
            primary=[.4,.6],secondary=[.3,.4],stress_sources=500,bootstrap=2000,bootstrap_seed=20261107)
PAIRS=((.3,.4),(.4,.3),(.4,.4),(.4,.5),(.4,.6),(.4,.7))
CONDITIONS=('clean','lr2','lr4','lr8','blur1','blur2','jpeg30','jpeg10')
PRIMARY=PAIRS.index((.4,.6))
