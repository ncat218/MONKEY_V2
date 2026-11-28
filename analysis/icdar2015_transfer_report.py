"""Rebuild the Vietnamese ICDAR transfer report from published per-image counts.

No model inference, fitting, threshold selection or dataset access.
Run: python analysis/icdar2015_transfer_report.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'docs/handoff/artifacts/icdar2015_192199'
OUTPUT = ROOT/'docs/handoff/results/icdar2015_transfer_192199.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    result = json.loads((INPUT/'icdar2015_summary.json').read_text())
    complete = json.loads((INPUT/'complete.json').read_text())
    assert sha(INPUT/'icdar2015_report.md') == complete['report_sha256']
    assert sha(INPUT/'icdar2015_summary.json') == complete['summary_sha256']
    audit = json.loads((INPUT/'b0_per_image_metric_audit.json').read_text())
    assert audit['per_image_count_differences'] == audit['matched_gt_differences'] == []
    scores = result['scores']
    data = np.load(INPUT/'source_counts.npz')
    get = lambda c: data[c.replace('/','__')]
    for cell, v in scores.items():
        m = get(cell)
        assert m.shape == (500,11) and np.array_equal(m[:,1],get('B0/reproduce')[:,1])
        assert np.array_equal(m[:,[3,5,7,9]], get('B0/reproduce')[:,[3,5,7,9]])
        tp,gt,pred = m.sum(axis=0)[:3]
        assert gt == 2077 and abs(2*tp/(gt+pred)-v['hmean']) < 1e-12
        assert np.array_equal(m[:,[3,5,7,9]].sum(axis=0),[0,137,826,1114])
    draws = np.random.default_rng(20261107).integers(500,size=(2000,500))
    def metrics(v):
        v = np.asarray(v,dtype=float)
        with np.errstate(divide='ignore',invalid='ignore'):
            return np.stack([200*v[...,0]/(v[...,1]+v[...,2]),100*v[...,0]/v[...,2],
                             100*v[...,0]/v[...,1],100*(v[...,4]+v[...,6])/(v[...,3]+v[...,5]),
                             v[...,2]-v[...,0],v[...,1]-v[...,0]],axis=-1)
    point = {c:metrics(get(c).sum(axis=0)) for c in scores}
    bootstrap = {c:metrics(get(c)[draws].sum(axis=1)) for c in scores}
    pairs = [
        ('ASF so B1','ASF/native/F','B1/native/F'),
        ('ASF so Nearest train tiếp','ASF/native/F','Nearest/native/F'),
        ('EDSR+B1 so B1','B1/edsr_x2/F','B1/native/F'),
        ('EDSR+B1 so bicubic+B1','B1/edsr_x2/F','B1/bicubic_x2/F'),
        ('ASF+EDSR so EDSR+B1','ASF/edsr_x2/F','B1/edsr_x2/F'),
        ('Thêm Q trên ASF+EDSR','ASF/edsr_x2/Q','ASF/edsr_x2/F'),
        ('Tổ hợp Q so B1','ASF/edsr_x2/Q','B1/native/F'),
        ('Tổ hợp Q so B-Clean','ASF/edsr_x2/Q','B-Clean/native/F'),
        ('Tổ hợp Q so B0 reproduce','ASF/edsr_x2/Q','B0/reproduce'),
    ]
    contrasts = {}
    keys = ('hmean_pp','precision_pp','recall_pp','small_lt16_recall_pp','fp_count','fn_count')
    for label,new,old in pairs:
        delta = point[new]-point[old]
        interval = np.quantile(bootstrap[new]-bootstrap[old],[.025,.975],axis=0)
        contrasts[label] = dict(new=new,old=old,metrics={k:dict(delta=float(delta[i]),ci95=interval[:,i].tolist())
                                                       for i,k in enumerate(keys)})
        if old in result['comparisons']:
            expected = result['comparisons'][old][new]['hmean_pp']['ci95']
            assert np.allclose(interval[:,0],expected,atol=1e-10,rtol=0)
    meta = dict(method='2000 paired-image bootstrap, seed20261107; same draws as frozen worker',
                source_counts_sha256=sha(INPUT/'source_counts.npz'),no_inference=True,no_fit=True,
                contrasts=contrasts)
    (INPUT/'component_contrasts.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    labels = [('B0 — setting reproduce','B0/reproduce'),('B0 — setting chung `.40/.60`','B0/common'),
              ('B-Clean — fine-tune 100% Clean','B-Clean/native/F'),('B1 — fine-tune 50/50 Clean/LR2','B1/native/F'),
              ('B1 + ASF','ASF/native/F'),('B1 + EDSR ×2','B1/edsr_x2/F'),
              ('B1 + ASF + EDSR ×2','ASF/edsr_x2/F'),('B1 + ASF + EDSR ×2 + Q','ASF/edsr_x2/Q')]
    lines = ['# ICDAR2015 gốc: kết quả các pipeline đã khóa — job 192199','',
        'Ngày 28/11/2026. Job **COMPLETED, exit 0:0, 57 phút 18 giây**, H200 `h200node01`, MIG 2g.35gb.',
        '**Kết luận:** tổ hợp ASF+EDSR+Q cải thiện Hmean so B1 đã fine-tune, nhưng chưa vượt B0 hay B-Clean trên ICDAR2015 gốc. Không có mức tăng Recall chữ nhỏ của tổ hợp cuối so B1 trong phép đo này.','',
        '## 1. Đã kiểm tra đúng bộ reproduce','',
        '- Đúng 500 ảnh gốc và manifest MMOCR; SHA256 ZIP/annotation/audit và cả 500 ảnh đều khớp. Kích thước, EXIF đều PASS.',
        '- Có 2.077 GT hợp lệ, 3.153 GT ignore. Không tạo LR2 cho bảng này.',
        '- B0 setting reproduce tái hiện TP1.800/FP178/FN277, Hmean88,7793%.',
        '- Kiểm tra thêm raw predictions của job11339: **500/500 ảnh khớp TP/FP/FN và danh sách GT được match**. Một số score/quads không giống tuyệt đối về số học; không tuyên bố prediction bit-identical.',
        '- Code, config, dữ liệu, 5 checkpoint detector, EDSR và evaluator giữ nguyên từ lúc submit đến cuối run. 6 kiểm thử CPU PASS trước inference.',
        '- Chỉ inference; không train, chọn checkpoint, fit Q hay tune ngưỡng trên ICDAR test.','',
        '## 2. Kết quả chính','',
        '| Pipeline | P (%) | R (%) | Hmean (%) | R chữ <16 px (%) | FP | FN | Median ms/ảnh |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,cell in labels:
        v = scores[cell]
        small = sum(b['tp'] for b in v['bins'][:2])/137
        lines.append(f'| {label} | {100*v["precision"]:.2f} | {100*v["recall"]:.2f} | {100*v["hmean"]:.2f} | {100*small:.2f} | {v["fp"]} | {v["fn"]} | {result["latency"][cell]["median_ms"]:.1f} |')
    lines += ['', '**Chữ nhỏ:** chiều cao là y-extent polygon GT ở ảnh gốc, giữ nguyên trước mọi phép phóng. Nhóm <8 px có **0 GT**, nên tiny recall là **N/A**, không phải 0%. Nhóm <16 px có **137 GT**; các nhóm 16–<32 và ≥32 có 826 và 1.114 GT. Không đồng nhất nhóm này với tiny <4 px trên native LR2 của TextOCR.', '',
        '**Setting:** B0 reproduce dùng mask `.30`, epsilon `.01`, score native `.30` rồi packed score cutoff `.60` đã khóa lịch sử. Hàng B0 common và mọi F dùng `.40/.60`, epsilon `.01`; B0 frozen riêng trong bảng đầy đủ dùng `.30/.45`, epsilon `.005`, đã chọn từ TextOCR. Q dùng raw pool ≥`.20`, cutoff quality `.3315908613227198` đã khóa trên TRAIN. Không chọn setting tốt nhất từ bảng ICDAR.', '',
        '## 3. CI ghép cặp và đóng góp từng thành phần','',
        '| Phép so (mới − cũ) | ΔHmean pp [95% CI] | ΔRecall pp [95% CI] | ΔR <16 px pp [95% CI] | ΔFP | ΔFN |',
        '|---|---:|---:|---:|---:|---:|']
    for label,new,old in pairs:
        m = contrasts[label]['metrics']
        def fmt(key):
            v=m[key];lo,hi=v['ci95'];return f'{v["delta"]:+.3f} [{lo:+.3f}; {hi:+.3f}]'
        lines.append(f'| {label} | {fmt("hmean_pp")} | {fmt("recall_pp")} | {fmt("small_lt16_recall_pp")} | {m["fp_count"]["delta"]:+.0f} | {m["fn_count"]["delta"]:+.0f} |')
    lines += ['', '2.000 bootstrap theo **ảnh**, seed20261107; khớp các CI đã xuất bởi worker. Không bootstrap từng từ như các mẫu độc lập. CI không đo biến thiên seed train; đây là các checkpoint seed42, benchmark đã có lịch sử xem lúc reproduce.', '',
        '## 4. Diễn giải','',
        '1. **Cải thiện trên B1 có chuyển được một phần:** EDSR+B1 tăng Hmean 75,80→78,36%, FP887→689; Recall giảm 87,10→85,80%, FN268→295. Lợi ích tổng thể đi kèm giảm FP, không chứng minh phục hồi thêm chữ nhỏ ở ảnh ICDAR gốc.',
        '2. **ASF chưa cho thấy đóng góp Hmean riêng rõ:** ASF native76,04% so Nearest76,01%. Thêm ASF khi đã có EDSR cho78,22% so EDSR+B1 78,36%. Xem CI để tránh quy tác dụng train thêm cho ASF.',
        '3. **Q không lặp lại lợi ích TextOCR:** trên ASF+EDSR, thêm Q cho Hmean78,09% so78,22%; FP691→700, FN299→298. Đây là point estimate; CI trong bảng cho biết độ bất định. Không fit hoặc chọn lại cutoff để cứu kết quả test.',
        '4. **B0 vẫn mạnh nhất ở benchmark gốc.** Tổ hợp Q thua B0 reproduce khoảng10,68pp Hmean; vẫn thua B0 common `.40/.60`. Chênh lệch không chỉ do đối chiếu hai điểm vận hành khác nhau. B-Clean82,20% cũng cao hơn các nhánh B1 trong lượt này.',
        '5. **Không ngoại suy từ LR2 TextOCR sang mọi ảnh.** Trên development TextOCR LR2, tổ hợp Q từng tăng Hmean/tiny recall; trên ICDAR gốc, mức tăng so B1 chủ yếu thể hiện ở Precision/FP và chưa giữ được năng lực B0. Hai kết quả không được gộp thành tuyên bố cải thiện phổ quát.',
        '6. **Chi phí:** B1 median177,7ms; tổ hợp Q503,0ms trong audit20 ảnh/3 warmup/2 lượt, GPU dùng chung. Tính SR+detector/postprocessing; không tính ghi PNG, hash, metric hoặc kiểm tra parity lặp. Không phải cam kết latency production.','',
        '## 5. Đối chiếu paper và phạm vi kết luận','',
        'Paper §4.3 ghi DBNet+MonkeyOCRv2 **88,5%**; README detection chính thức ghi **88,9%**. Giữ riêng hai con số công bố; neo đối chứng của lượt này là **B0 FP32 tái lập88,78%** và thêm hàng B0 common cùng ngưỡng. Đây là tái lập suy luận từ checkpoint phát hành, không phải tái huấn luyện toàn bộ paper.',
        '[Paper chính thức](https://arxiv.org/html/2607.11562v1#S4.SS3), [README detection chính thức](https://github.com/Yuliang-Liu/MonkeyOCRv2/tree/main/detection/mmocr).','',
        'B0 từng train trên ICDAR TRAIN; 500 ảnh test đã xem trong reproduce và đánh giá trước đây. Lượt này đo **chuyển giao trở lại benchmark gốc sau thích nghi TextOCR**, không phải zero-shot trên một holdout chưa từng thấy. Chỉ dùng kết quả để mô tả chuyển giao/giới hạn, không chọn lại phương pháp hoặc ngưỡng.', '',
        'Báo cáo đồ án nên giữ LR2 TextOCR là bài toán chính, trình bày cải tiến so B1 cùng đối chứng B-Clean và ablation. ICDAR2015 là bằng chứng về **giới hạn tổng quát hóa/giữ năng lực benchmark gốc**, không phải kết quả vượt paper. Nguyên nhân FP cụ thể cần review ảnh và matching; chưa khẳng định mọi FP là nền hay lỗi nhãn.','',
        '## 6. File và bằng chứng','',
        '- [Bảng đủ 12 setting và CI gốc](../artifacts/icdar2015_192199/icdar2015_report.md).',
        '- [Summary JSON đầy đủ](../artifacts/icdar2015_192199/icdar2015_summary.json), [counts500 ảnh/cell](../artifacts/icdar2015_192199/source_counts.npz), [CI thành phần](../artifacts/icdar2015_192199/component_contrasts.json).',
        '- [Audit dữ liệu](../artifacts/icdar2015_192199/data_audit.json), [B0 từng ảnh](../artifacts/icdar2015_192199/b0_per_image_metric_audit.json), [identity cuối](../artifacts/icdar2015_192199/final_identity_audit.json), [Slurm receipt](../artifacts/icdar2015_192199/slurm_job.txt).',
        '- [Index đúng500 ảnh](../data/icdar2015_reproduce/source_ids.csv); index không thay thế GT polygons.',
        '- [Spec khóa trước inference](../../icdar2015-transfer-v1.md), [launcher](../../../submit_icdar_transfer.py).',
        '- Rebuild bảng/CI: `python analysis/icdar2015_transfer_report.py` (chỉ đọc artifacts; không inference/train).','',
        'Dữ liệu gốc còn trên HPC tại `/datastore/dungmt/textvqa/Tuan/monkeyocr_workspace/ICDAR2015__Monkey_E0_Aprime_FP32_transfer/inputs/` gồm `ch4_test_images.zip`, `dataset_audit.json`, `textdet_test__recovered_byte_identical_to_Phase1.json`. Manifest cuối là MMOCR JSON có `metainfo`/`data_list`, 500 hàng ảnh và `instances` chứa polygon/bbox/ignore. SHA256 annotation `fd1a9d52dbd2b3ed86a16fef27cda35e4c44624d07c4f1c25338ea2ca660f718`.',
        'Raw predictions theo từng ảnh vẫn được giữ tại `.runtime/icdar2015_transfer/v1/records/` trên HPC. Không phân phối lại ảnh trong Git.']
    OUTPUT.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('PASS: all12 counts/bins, completion hashes, per-image B0 audit and bootstrap reproduction.')
    for label,new,old in pairs:
        print(label,contrasts[label]['metrics']['hmean_pp'])
    print(OUTPUT)


if __name__ == '__main__': run()
