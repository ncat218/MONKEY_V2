import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DEFAULT_PARAMS = {'motion_text': {'L1': [0.1, 0.2], 'L2': [0.2, 0.4], 'L3': [0.4, 0.6]}, 'motion_fallback': {'L1': [0.004, 0.008], 'L2': [0.008, 0.016], 'L3': [0.016, 0.028]}, 'motion_max': {'L1': 7, 'L2': 13, 'L3': 21}, 'defocus_text': {'L1': [0.04, 0.08], 'L2': [0.08, 0.16], 'L3': [0.16, 0.25]}, 'defocus_fallback': {'L1': [0.0015, 0.0035], 'L2': [0.0035, 0.007], 'L3': [0.007, 0.012]}, 'defocus_max': {'L1': 2, 'L2': 4, 'L3': 6}, 'low_resolution_scale': {'L1': [0.7, 0.9], 'L2': [0.4, 0.7], 'L3': [0.25, 0.45]}, 'projected_text_guard': {'L1': 7, 'L2': 5, 'L3': 3}, 'lowlight_ev': {'L1': [-1, -0.5], 'L2': [-2, -1], 'L3': [-3, -2]}, 'lowlight_gamma': {'L1': [1, 1.15], 'L2': [1.05, 1.3], 'L3': [1.1, 1.5]}, 'lowlight_peak': {'L1': [120, 250], 'L2': [50, 120], 'L3': [20, 60]}, 'lowlight_read': {'L1': [0.002, 0.006], 'L2': [0.005, 0.015], 'L3': [0.01, 0.03]}, 'overexposure_ev': {'L1': [0.3, 0.8], 'L2': [0.8, 1.5], 'L3': [1.5, 2.2]}, 'glare_probability': {'L1': 0.25, 'L2': 0.5, 'L3': 0.7}, 'glare_alpha': {'L1': [0.05, 0.12], 'L2': [0.1, 0.25], 'L3': [0.2, 0.4]}, 'perspective_displacement': {'L1': [0.01, 0.025], 'L2': [0.025, 0.06], 'L3': [0.06, 0.1]}, 'jpeg_quality': {'L1': [70, 90], 'L2': [40, 70], 'L3': [20, 45]}}
P = DEFAULT_PARAMS

# Original notebook cell below, byte-for-byte unchanged.
# Degradation operators. Every random draw comes from a SHA-256-derived per-image generator.
SINGLES = ('motion_camera_shake','defocus_blur','lowlight_sensor_noise','overexposure_glare','low_resolution','perspective','jpeg_compression')
ORDER = {'perspective':0,'motion_camera_shake':1,'defocus_blur':1,'lowlight_sensor_noise':2,'overexposure_glare':2,'low_resolution':3,'jpeg_compression':4}

def draw(rng, key, level):
    lo,hi = P[key][level]
    return float(rng.uniform(lo,hi))

def odd_clamp(value, maximum):
    n = int(round(value)); n = max(3,min(maximum,n))
    if n % 2 == 0: n = n+1 if n < maximum else n-1
    return n

def motion_psf(length, angle, shake, rng):
    k = np.zeros((length,length),np.float32)
    if shake:
        x = y = (length-1)/2
        points = [(x,y)]
        direction = np.deg2rad(angle)
        for _ in range(max(3,length*4)):
            direction += rng.normal(0,.20)
            x += .25*np.cos(direction); y += .25*np.sin(direction)
            points.append((x,y))
        xy = np.asarray(points); xy -= xy.mean(axis=0); xy += (length-1)/2
        pts = np.rint(np.clip(xy,0,length-1)).astype(int)
        for x,y in pts: k[y,x] += 1
    else:
        center = (length-1)/2; radius = center
        x0,y0 = center-radius*np.cos(np.deg2rad(angle)),center-radius*np.sin(np.deg2rad(angle))
        x1,y1 = center+radius*np.cos(np.deg2rad(angle)),center+radius*np.sin(np.deg2rad(angle))
        cv2.line(k,(round(x0),round(y0)),(round(x1),round(y1)),1,1)
    k /= k.sum()
    return k

def apply_op(img, op, level, h_text, rng):
    h,w = img.shape[:2]; d = min(h,w); par = {}
    if op == 'motion_camera_shake':
        raw = draw(rng, 'motion_text' if h_text else 'motion_fallback', level)*(h_text if h_text else d)
        length = odd_clamp(raw,P['motion_max'][level]); angle = float(rng.uniform(0,180)); shake = bool(rng.integers(0,2))
        kernel = motion_psf(length,angle,shake,rng)
        img = cv2.filter2D(img,-1,kernel,borderType=cv2.BORDER_REFLECT101)
        par = dict(length_px=length,angle_deg=angle,psf='random_walk' if shake else 'linear')
    elif op == 'defocus_blur':
        raw = draw(rng,'defocus_text' if h_text else 'defocus_fallback',level)*(h_text if h_text else d)
        radius = int(np.clip(round(raw),1,P['defocus_max'][level])); k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)).astype(np.float32); k /= k.sum()
        img = cv2.filter2D(img,-1,k,borderType=cv2.BORDER_REFLECT101); par = dict(radius_px=radius)
    elif op == 'lowlight_sensor_noise':
        ev = draw(rng,'lowlight_ev',level); gamma = draw(rng,'lowlight_gamma',level)
        peak = draw(rng,'lowlight_peak',level); read = draw(rng,'lowlight_read',level)
        x = np.clip((img.astype(np.float32)/255)*(2**ev),0,1)**gamma
        img = np.rint(np.clip(x,0,1)*255).astype(np.uint8)
        par = dict(ev=ev,gamma=gamma,poisson_peak=peak,read_std=read)
    elif op == 'overexposure_glare':
        ev = draw(rng,'overexposure_ev',level)
        x = np.clip(img.astype(np.float32)/255*(2**ev),0,1)
        glare = bool(rng.random() < P['glare_probability'][level]); par = dict(ev=ev,glare=glare)
        if glare:
            alpha = draw(rng,'glare_alpha',level); cx=float(rng.uniform(0,w)); cy=float(rng.uniform(0,h)); sigma=float(rng.uniform(.20,.50)*d)
            yy,xx=np.ogrid[:h,:w]; mask=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma*sigma)).astype(np.float32)
            x = x+alpha*mask[...,None]*(1-x); par.update(alpha=alpha,center_x=cx,center_y=cy,sigma_px=sigma)
        img=np.rint(np.clip(x,0,1)*255).astype(np.uint8)
    elif op == 'low_resolution':
        scale=draw(rng,'low_resolution_scale',level); adjusted=False
        if h_text and h_text*scale < P['projected_text_guard'][level]:
            scale=min(1.0,P['projected_text_guard'][level]/h_text); adjusted=True
        sw,sh=max(1,round(w*scale)),max(1,round(h*scale))
        small=cv2.resize(img,(sw,sh),interpolation=cv2.INTER_AREA)
        img=cv2.resize(small,(w,h),interpolation=cv2.INTER_CUBIC)
        par=dict(scale=scale,downsample_size=[sw,sh],scale_adjusted_for_text_guard=adjusted)
    elif op == 'perspective':
        magnitude=draw(rng,'perspective_displacement',level)*d
        src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        accepted=False
        for attempt in range(50):
            dst=src+rng.uniform(-magnitude,magnitude,(4,2)).astype(np.float32)
            M=cv2.getPerspectiveTransform(src,dst)
            valid=cv2.warpPerspective(np.ones((h,w),np.uint8),M,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            border=float(1-valid.mean())
            if np.isfinite(M).all() and border <= .05: accepted=True; break
        if not accepted: raise RuntimeError('Perspective transform exceeded 5% border in 50 attempts')
        img=cv2.warpPerspective(img,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
        par=dict(corners=dst.tolist(),border_fraction=border,attempts=attempt+1,magnitude_px=magnitude)
    elif op == 'jpeg_compression':
        lo,hi=P['jpeg_quality'][level]; quality=int(rng.integers(lo,hi+1))
        bgr=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        ok,buf=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,quality])
        assert ok
        decoded=cv2.imdecode(buf,cv2.IMREAD_COLOR)
        assert decoded is not None
        img=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
        par=dict(quality=quality)
    else: raise ValueError(op)
    assert img.shape==(h,w,3) and img.dtype==np.uint8
    return img,par

def add_sensor_noise(img, lowlight_params, rng):
    x=img.astype(np.float32)/255
    peak=lowlight_params['poisson_peak']; read=lowlight_params['read_std']
    x=rng.poisson(x*peak).astype(np.float32)/peak+rng.normal(0,read,x.shape).astype(np.float32)
    return np.rint(np.clip(x,0,1)*255).astype(np.uint8)

def choose_mix(level,rng):
    count = {'L1':2,'L2':int(rng.integers(2,4)),'L3':int(rng.integers(3,5))}[level]
    pool=list(SINGLES)
    while True:
        chosen=list(rng.choice(pool,size=count,replace=False))
        optical=sum(x in ('motion_camera_shake','defocus_blur') for x in chosen)
        illum=sum(x in ('lowlight_sensor_noise','overexposure_glare') for x in chosen)
        if optical <= 1 and illum <= 1 and any(x in chosen for x in ('motion_camera_shake','defocus_blur','lowlight_sensor_noise','overexposure_glare','low_resolution')): break
    chosen.sort(key=lambda x:ORDER[x])
    if 'perspective' in chosen and any(x in chosen for x in ('motion_camera_shake','defocus_blur')) and rng.random()<.5:
        chosen[0],chosen[1]=chosen[1],chosen[0]
    return chosen

def degrade(clean, condition, level, h_text, seed):
    rng=np.random.default_rng(seed)
    ops=choose_mix(level,rng) if condition=='realistic_mix' else [condition]
    out=clean.copy(); params=[]; pending_lowlight=None
    for op in ops:
        if op=='jpeg_compression' and pending_lowlight is not None:
            out=add_sensor_noise(out,pending_lowlight,rng); pending_lowlight=None
        out,p=apply_op(out,op,level,h_text,rng)
        if op=='lowlight_sensor_noise': p['sensor_noise_stage']='after_low_resolution' if 'low_resolution' in ops else 'after_illumination'
        params.append(dict(operation=op,**p))
        if op=='lowlight_sensor_noise': pending_lowlight=p
    if pending_lowlight is not None: out=add_sensor_noise(out,pending_lowlight,rng)
    return out,params

def diagnostics(clean,out):
    gray=cv2.cvtColor(out,cv2.COLOR_RGB2GRAY)
    psnr=float(peak_signal_noise_ratio(clean,out,data_range=255))
    return dict(psnr=psnr if np.isfinite(psnr) else None,
                ssim=float(structural_similarity(clean,out,channel_axis=2,data_range=255)),
                mean_luminance=float(gray.mean()),std_luminance=float(gray.std()),
                laplacian_variance=float(cv2.Laplacian(gray,cv2.CV_64F).var()))
