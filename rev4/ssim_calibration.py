"""Rev.4 TRAIN-only calibration helpers. Importing this file does not start calibration."""
from bisect import bisect_right
from collections import Counter
from fractions import Fraction
from hashlib import file_digest, sha256
from math import ceil, floor, isfinite
from pathlib import Path
from random import Random
import json
import unicodedata

CONDITIONS = ('Blur1', 'JPEG30')
BIN_STARTS = (.30, .40, .50, .60, .70, .80, .90)
QUOTAS = (20, 20, 20, 20, 20, 20, 80)
THRESHOLDS = tuple(value/100 for value in range(30, 91, 5))
TRAIN_NAME = 'manifest__train__textvqa_clean_degraded_same_textocr_word_gt.jsonl'
TRAIN_SHA256 = 'ca1b91636cbe025823041ce336ffb59affa826a073f666dd7ba386bcab6c983d'


def require_lr2_failure(gates):
    keys = ('G1', 'G2', 'G3')
    if any(type(gates.get(key)) is not bool for key in keys):
        raise ValueError('Require all three confirmed S2 LR2 gate outcomes')
    if all(gates[key] for key in keys):
        raise ValueError('Rev.4 skips blind calibration after LR2 passes all gates')


def canonical_ids(ids):
    ordered = sorted(set(ids))
    if any(not value or any(c in value for c in '\t\r\n') for pair in ordered for value in pair):
        raise ValueError('Invalid canonical source/word ID')
    return ''.join(f'{source}\t{word}\n' for source, word in ordered).encode('utf-8')


def pick_pool(word_ids):
    ordered = sorted(set(word_ids))
    return sorted(Random(20261109).sample(ordered, min(20_000, len(ordered))))


def load_train_pool(manifest, lr2_gates, id_file):
    """Lock TRAIN word IDs and return clean-image and polygon references for the same pool."""
    require_lr2_failure(lr2_gates)
    manifest = Path(manifest)
    if manifest.name != TRAIN_NAME:
        raise ValueError('Require the frozen TRAIN manifest filename')
    with manifest.open('rb') as handle:
        if file_digest(handle, 'sha256').hexdigest() != TRAIN_SHA256:
            raise ValueError('Require the frozen FULL_R1 TRAIN manifest SHA-256')
        handle.seek(0)
        pool = pick_pool((row['source_image_id'], str(ann['ann_id']))
                         for row in map(json.loads, handle)
                         for ann in row['gt_clean']['valid'])
        selected = set(pool)
        refs = {}
        handle.seek(0)
        for row in map(json.loads, handle):
            source = row['source_image_id']
            for ann in row['gt_clean']['valid']:
                key = source, str(ann['ann_id'])
                if key in selected:
                    refs[key] = (row['clean_relpath'], row['clean_sha256'], ann['geometry'])
    payload = canonical_ids(pool)
    with Path(id_file).open('xb') as sink:
        sink.write(payload)
    return dict(pool=pool, refs=refs, id_sha256=sha256(payload).hexdigest())


def word_crops(clean, degraded, geometry):
    """Crop both images using the clean GT axis-aligned bounds."""
    from shapely.geometry import shape

    if clean.size != degraded.size:
        raise ValueError('Blur/JPEG crop inputs require identical canvas dimensions')
    width, height = clean.size
    x0, y0, x1, y1 = shape(geometry).bounds
    if not all(isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    box = max(0, floor(x0)), max(0, floor(y0)), min(width, ceil(x1)), min(height, ceil(y1))
    if min(box[2]-box[0], box[3]-box[1]) < 7:
        return None
    return clean.crop(box), degraded.crop(box)


def crop_ssim(clean, degraded, geometry):
    import numpy as np
    import PIL
    from skimage.metrics import structural_similarity

    if PIL.__version__ != '11.3.0':
        raise ValueError('Require Pillow 11.3.0 for calibration crops')
    crops = word_crops(clean, degraded, geometry)
    if crops is None:
        return None
    a, b = (np.asarray(image.convert('L')) for image in crops)
    score = float(structural_similarity(a, b, data_range=255, win_size=7))
    return score if isfinite(score) else None


def score_pool(pool, get_pair, lr2_gates):
    """get_pair(condition, word_id) supplies verified clean/degraded images and clean GT."""
    require_lr2_failure(lr2_gates)
    if list(pool) != sorted(set(pool)) or not 0 < len(pool) <= 20_000:
        raise ValueError('Require a unique, sorted TRAIN word pool of at most 20,000')
    return {condition: {word: crop_ssim(*get_pair(condition, word)) for word in pool}
            for condition in CONDITIONS}


def bin_index(value):
    if value is None or not isfinite(value) or not .30 <= value <= 1.00:
        return None
    return bisect_right(BIN_STARTS, value)-1


def population(scores):
    bins = [[] for _ in BIN_STARTS]
    counts = Counter(dict(invalid=0, below_0_30=0, below_zero=0, above_one=0))
    for word, value in sorted(scores.items()):
        index = bin_index(value)
        if index is not None:
            bins[index].append(word)
        elif value is None or not isfinite(value):
            counts['invalid'] += 1
        elif value < .30:
            counts['below_0_30'] += 1
            counts['below_zero'] += int(value < 0)
        else:
            counts['above_one'] += 1
    return bins, dict(counts)


def select_cards(scores_by_condition):
    """Sample within a common pool, preferring disjoint JPEG IDs when possible."""
    if set(scores_by_condition) != set(CONDITIONS):
        raise ValueError('Require both Blur1 and JPEG30 score tables')
    if set(scores_by_condition['Blur1']) != set(scores_by_condition['JPEG30']):
        raise ValueError('Conditions must use exactly the same TRAIN pool')
    if not 0 < len(scores_by_condition['Blur1']) <= 20_000:
        raise ValueError('TRAIN pool must have between 1 and 20,000 words')
    results = {}
    prior = set()
    for condition in CONDITIONS:
        bins, exclusions = population(scores_by_condition[condition])
        sizes = [len(words) for words in bins]
        total = sum(sizes)
        if not total:
            results[condition] = dict(status='FAIL_NO_ELIGIBLE_CROPS', selected=[], exclusions=exclusions,
                                      population_by_bin=sizes, p_b=[0.0]*len(bins), overlap=0)
            continue
        rng = Random(20261108)
        chosen = set()

        def take(words, amount):
            candidates = [word for word in words if word not in chosen]
            preferred = [word for word in candidates if word not in prior]
            shared = [word for word in candidates if word in prior]
            take_preferred = min(amount, len(preferred))
            chosen.update(rng.sample(preferred, take_preferred))
            chosen.update(rng.sample(shared, min(amount-take_preferred, len(shared))))

        for words, quota in zip(bins, QUOTAS):
            take(words, quota)
        for words in reversed(bins):
            take(words, 200-len(chosen))
        selected = sorted(chosen)
        payload = canonical_ids(selected)
        results[condition] = dict(status='READY', selected=selected, selection_sha256=sha256(payload).hexdigest(),
                                  population_by_bin=sizes, p_b=[n/total for n in sizes],
                                  exclusions=exclusions, overlap=len(chosen & prior))
        prior.update(chosen)
    return results


def blind_layout(selections, scores, ground_truth, *, seed):
    """Return a public card index and a separate private key."""
    cards = []
    for condition in CONDITIONS:
        for word in selections[condition]['selected']:
            transcript = ground_truth[word]
            if not normalize(transcript):
                raise ValueError('Missing valid TRAIN transcript')
            for version in ('clean', 'degraded'):
                cards.append(dict(condition=condition, word=word, version=version,
                                  transcript=transcript, ssim=scores[condition][word]))
    Random(seed).shuffle(cards)
    private = {f'C{index:04d}': item for index, item in enumerate(cards, 1)}
    public = [{'card_id': card, 'image': f'{card}.png'} for card in private]
    return public, private


def save_blind_cards(public, private, get_crop, output):
    """Only share the cards directory. get_crop receives the private card metadata."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    cards = root/'cards'
    cards.mkdir()
    for item in public:
        card = item['card_id']
        crop = get_crop(private[card]).convert('RGB')
        crop.info.clear()
        crop.save(cards/item['image'], format='PNG')
    (cards/'index.json').write_text(json.dumps(public, ensure_ascii=False)+'\n', encoding='utf-8')
    (root/'private_key.json').write_text(json.dumps(private, ensure_ascii=False)+'\n', encoding='utf-8')


def normalize(text):
    return unicodedata.normalize('NFC', text).lower().strip()


def match_two_readers(private, reader_a, reader_b):
    """Evaluate two complete, independently collected card transcripts."""
    if any(set(reader) != set(private) or any(not isinstance(value, str) for value in reader.values())
           for reader in (reader_a, reader_b)):
        raise ValueError('Both readers must provide one text response for every card')
    outcomes = {condition: {} for condition in CONDITIONS}
    agreement = Counter()
    totals = Counter()
    for card, item in private.items():
        condition, word, version = item['condition'], tuple(item['word']), item['version']
        a, b, truth = map(normalize, (reader_a[card], reader_b[card], item['transcript']))
        record = outcomes[condition].setdefault(word, {})
        if version in record:
            raise ValueError('Duplicate condition/word/version in private key')
        record[version] = a == truth and b == truth
        totals[condition, version] += 1
        agreement[condition, version] += a == b
    if any(set(pair) != {'clean', 'degraded'} for rows in outcomes.values() for pair in rows.values()):
        raise ValueError('Missing a clean/degraded card for a selected word')
    rates = {condition: {version: dict(agree=agreement[condition, version], total=totals[condition, version])
                         for version in ('clean', 'degraded')} for condition in CONDITIONS}
    return outcomes, rates


def weighted_curve(scores, selected, outcomes):
    """Use only after both real readers return complete blind cards."""
    if len(selected) != len(set(selected)) or set(outcomes) != set(selected) or len(selected) > 200:
        raise ValueError('Outcomes must cover the locked card selection exactly')
    if any(set(pair) != {'clean', 'degraded'} or any(type(value) is not bool for value in pair.values())
           for pair in outcomes.values()):
        raise ValueError('Each selected word needs two boolean consensus outcomes')
    if any(word not in scores or bin_index(scores[word]) is None for word in selected):
        raise ValueError('All selected words must be eligible and in the TRAIN pool')
    bins, exclusions = population(scores)
    total = sum(map(len, bins))
    if not total:
        return dict(status='FAIL_NO_ELIGIBLE_CROPS', tau=None, exclusions=exclusions)
    members = [[] for _ in BIN_STARTS]
    for word in selected:
        members[bin_index(scores[word])].append(word)
    weights, by_bin = {}, []
    for lower, pop, read in zip(BIN_STARTS, bins, members):
        if pop and not read:
            raise ValueError(f'Occupied bin {lower:.2f} has no completed readings')
        p = Fraction(len(pop), total)
        weight = p/len(read) if read else None
        clean = sum(outcomes[word]['clean'] for word in read)
        both = sum(outcomes[word]['clean'] and outcomes[word]['degraded'] for word in read)
        by_bin.append(dict(lower=lower, p_b=float(p), n_b=len(read), clean_correct=clean,
                           correct_both=both, unweighted_rate=both/clean if clean else None,
                           weight=float(weight) if weight is not None else None,
                           status='observed' if pop else 'not observed in pool'))
        weights.update(dict.fromkeys(read, weight))
    curve = []
    for threshold in THRESHOLDS:
        clean = [word for word in selected if scores[word] >= threshold and outcomes[word]['clean']]
        both = [word for word in clean if outcomes[word]['degraded']]
        mass = sum(weights[word] for word in clean)
        eff = mass**2/sum(weights[word]**2 for word in clean) if mass else None
        weighted = sum(weights[word] for word in both)/mass if mass else None
        enough = len(clean) >= 30 and eff is not None and eff >= 30
        status = ('UNDERPOWERED' if not enough else
                  'PASS' if weighted >= Fraction(4, 5) else 'BELOW_READABILITY')
        curve.append(dict(t=threshold, N=len(clean), B=len(both), N_eff=float(eff) if eff is not None else None,
                          R_unweighted=len(both)/len(clean) if clean else None,
                          R_weighted=float(weighted) if weighted is not None else None, status=status))
    tau = None
    for row in reversed(curve):
        if row['status'] != 'PASS':
            break
        tau = row['t']
    return dict(status='PASS' if tau is not None else 'FAIL_NO_VALID_TAU', tau=tau, bins=by_bin,
                thresholds=curve, clean_incorrect=sum(not pair['clean'] for pair in outcomes.values()),
                exclusions=exclusions)
