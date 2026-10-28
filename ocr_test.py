import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from paddleocr import PaddleOCR

BASE = Path(__file__).parent
CARDS_DIR = BASE / "cards"
GT_DIR = BASE / "gt"
OUT_DIR = BASE / "out"

MODELS = {
    "korean": dict(lang="korean"),
    "default": dict(),
}

COMMON = dict(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LINE_TOLERANCE = 0.6
MATCH_THRESHOLD = 0.2

LANG_LABEL = {"ko": "한국어", "ja": "일본어", "zh": "중국어", "en": "영어", "?": "미분류"}
LANG_ORDER = ["ko", "ja", "zh", "en", "?"]
LANG_OVERRIDE = {}


def detect_language(text):
    has_hangul = has_kana = has_han = False

    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3 or 0x3130 <= code <= 0x318F:
            has_hangul = True
        elif 0x3040 <= code <= 0x30FF or ch == "〒":
            has_kana = True
        elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            has_han = True

    if has_hangul:
        return "ko"
    if has_kana:
        return "ja"
    if has_han:
        return "zh"
    return "en"


def group_into_lines(texts, boxes, tolerance=LINE_TOLERANCE):
    if not texts:
        return []

    items = []
    for text, box in zip(texts, boxes):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        items.append({
            "text": text,
            "x": x1,
            "y": (y1 + y2) / 2,
            "h": abs(y2 - y1),
        })

    heights = sorted(item["h"] for item in items)
    threshold = (heights[len(heights) // 2] or 1.0) * tolerance

    items.sort(key=lambda item: item["y"])

    groups = [[items[0]]]
    for item in items[1:]:
        if abs(item["y"] - groups[-1][-1]["y"]) <= threshold:
            groups[-1].append(item)
        else:
            groups.append([item])

    lines = []
    for group in groups:
        group.sort(key=lambda item: item["x"])
        lines.append(" ".join(item["text"] for item in group))
    return lines


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def squeeze(text):
    return "".join(text.split())


def cer(truth, pred):
    if not truth:
        return 0.0 if not pred else 1.0
    return levenshtein(truth, pred) / len(truth)


def line_recall(gt_lines, ocr_lines):
    targets = [line for line in gt_lines if squeeze(line)]
    if not targets:
        return 0.0

    matched = 0
    for gt_line in targets:
        best = min((cer(squeeze(gt_line), squeeze(line)) for line in ocr_lines),
                   default=1.0)
        matched += best <= MATCH_THRESHOLD
    return matched / len(targets)


def infix_distance(needle, haystack):
    if not needle:
        return 0
    if not haystack:
        return len(needle)

    previous = [0] * (len(haystack) + 1)
    for ca in needle:
        current = [previous[0] + 1]
        for j, cb in enumerate(haystack, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return min(previous)


def content_coverage(gt_lines, ocr_lines):
    targets = [line for line in gt_lines if squeeze(line)]
    if not targets:
        return 0.0, []

    haystack = squeeze(" ".join(ocr_lines))

    results = []
    for gt_line in targets:
        needle = squeeze(gt_line)
        score = infix_distance(needle, haystack) / len(needle)
        results.append((score <= MATCH_THRESHOLD, gt_line, score))

    covered = sum(1 for ok, *_ in results if ok)
    return covered / len(targets), results


def extract(result):
    data = result.json
    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    return (
        list(data.get("rec_texts", [])),
        np.asarray(data.get("rec_scores", [])).tolist(),
        np.asarray(data.get("rec_boxes", [])).tolist(),
    )


def run_model(engine, image_path):
    texts, scores, boxes, objects = [], [], [], []
    for result in engine.predict(str(image_path)):
        t, s, b = extract(result)
        texts += t
        scores += s
        boxes += b
        objects.append(result)

    return {
        "texts": texts,
        "scores": scores,
        "boxes": boxes,
        "objects": objects,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "lines": group_into_lines(texts, boxes),
    }


def save_visualization(objects):
    target = OUT_DIR / "images"
    target.mkdir(parents=True, exist_ok=True)
    for obj in objects:
        try:
            obj.save_to_img(str(target))
        except Exception as error:
            print(f"        [경고] 이미지 저장 실패: {error}")


def evaluate(stem, lines):
    gt_file = GT_DIR / f"{stem}.txt"
    if not gt_file.exists():
        return None

    gt_text = gt_file.read_text(encoding="utf-8")
    gt_lines = [line for line in gt_text.splitlines() if line.strip()]

    coverage, details = content_coverage(gt_lines, lines)
    return {
        "accuracy": 1 - cer(squeeze(gt_text), squeeze("\n".join(lines))),
        "recall": line_recall(gt_lines, lines),
        "coverage": coverage,
        "misses": [d for d in details if not d[0]],
    }


def save(stem, image_name, language, chosen, results):
    OUT_DIR.mkdir(exist_ok=True)
    best = results[chosen]

    payload = {
        "image": image_name,
        "language": language,
        "chosen_model": chosen,
        "avg_score": best["avg_score"],
        "lines": best["lines"],
        "raw": {
            name: {k: r[k] for k in ("texts", "scores", "boxes", "avg_score")}
            for name, r in results.items()
        },
    }
    (OUT_DIR / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / f"{stem}.txt").write_text(
        "\n".join(best["lines"]), encoding="utf-8")


def width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text, size, align="left"):
    space = " " * max(0, size - width(text))
    return space + text if align == "right" else text + space


def percent(value):
    return f"{value * 100:.1f}%" if value is not None else "-"


def print_card(index, total, image_name, language, results, chosen, report):
    print(f"[{index}/{total}] {image_name}  ({LANG_LABEL[language]})")

    for name, result in results.items():
        mark = "←" if name == chosen else " "
        print(f"        {pad(name, 9)}신뢰도 {result['avg_score']:.3f}"
              f"   {len(result['texts']):>2}개 인식  {mark}")

    if report:
        print(f"        내용 포함률 {percent(report['coverage'])}"
              f"   문자 정확도 {percent(report['accuracy'])}"
              f"   줄 일치율 {percent(report['recall'])}")
        for _, gt_line, score in report["misses"]:
            print(f"          X  {gt_line}   (오류 {score * 100:.0f}%)")
    else:
        print("        정답지 없음")
    print()


def print_language_summary(rows):
    print("=" * 72)
    print("언어별 요약\n")
    header = (f"  {pad('언어', 10)}{pad('장수', 7, 'right')}{pad('측정', 7, 'right')}"
              f"{pad('내용 포함률', 16, 'right')}{pad('문자 정확도', 16, 'right')}"
              f"{pad('줄 일치율', 14, 'right')}")
    print(header)
    print("  " + "-" * 66)

    def average(group, key):
        measured = [r for r in group if r["report"]]
        if not measured:
            return None
        return sum(r["report"][key] for r in measured) / len(measured)

    def row_text(label, group):
        measured = [r for r in group if r["report"]]
        return (f"  {pad(label, 10)}"
                f"{pad(str(len(group)), 7, 'right')}"
                f"{pad(str(len(measured)), 7, 'right')}"
                f"{pad(percent(average(group, 'coverage')), 16, 'right')}"
                f"{pad(percent(average(group, 'accuracy')), 16, 'right')}"
                f"{pad(percent(average(group, 'recall')), 14, 'right')}")

    for language in LANG_ORDER:
        group = [r for r in rows if r["language"] == language]
        if group:
            print(row_text(LANG_LABEL[language], group))

    print("  " + "-" * 66)
    print(row_text("전체", rows))
    print()


def print_model_summary(rows):
    print("모델 채택 현황\n")
    for name in MODELS:
        chosen = [r for r in rows if r["chosen"] == name]
        if not chosen:
            continue
        languages = sorted({LANG_LABEL[r["language"]] for r in chosen})
        print(f"  {pad(name, 10)}{len(chosen):>2}장   {', '.join(languages)}")
    print()


def print_card_table(rows):
    print("명함별 결과\n")
    name_width = max([width(r["image"]) for r in rows] + [10]) + 2
    print(f"  {pad('파일', name_width)}{pad('언어', 10)}{pad('모델', 10)}"
          f"{pad('신뢰도', 10, 'right')}{pad('포함률', 10, 'right')}"
          f"{pad('정확도', 10, 'right')}{pad('줄일치', 10, 'right')}")
    print("  " + "-" * (name_width + 60))

    for row in rows:
        report = row["report"]
        print(f"  {pad(row['image'], name_width)}"
              f"{pad(LANG_LABEL[row['language']], 10)}"
              f"{pad(row['chosen'], 10)}"
              f"{pad('%.3f' % row['avg'], 10, 'right')}"
              f"{pad(percent(report['coverage'] if report else None), 10, 'right')}"
              f"{pad(percent(report['accuracy'] if report else None), 10, 'right')}"
              f"{pad(percent(report['recall'] if report else None), 10, 'right')}")
    print()


def main():
    if not CARDS_DIR.exists():
        print(f"[오류] {CARDS_DIR} 폴더가 없습니다.")
        sys.exit(1)

    images = sorted(p for p in CARDS_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"[오류] {CARDS_DIR} 안에 이미지가 없습니다.")
        sys.exit(1)

    print(f"명함 {len(images)}장 · 모델 {len(MODELS)}개\n")
    print("모델 로딩 중...")
    engines = {name: PaddleOCR(**COMMON, **kwargs) for name, kwargs in MODELS.items()}
    print("완료\n")

    rows = []
    for index, image in enumerate(images, start=1):
        results = {name: run_model(engine, image) for name, engine in engines.items()}
        chosen = max(results, key=lambda name: results[name]["avg_score"])
        best = results[chosen]

        language = LANG_OVERRIDE.get(image.stem) or detect_language(" ".join(best["texts"]))
        report = evaluate(image.stem, best["lines"])

        save(image.stem, image.name, language, chosen, results)
        save_visualization(best["objects"])
        print_card(index, len(images), image.name, language, results, chosen, report)

        rows.append({
            "image": image.name,
            "language": language,
            "chosen": chosen,
            "avg": best["avg_score"],
            "report": report,
        })

    print_language_summary(rows)
    print_model_summary(rows)
    print_card_table(rows)
    print(f"결과 저장: {OUT_DIR}")


if __name__ == "__main__":
    main()