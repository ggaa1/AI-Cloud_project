import argparse
import contextlib
import json
import os
import sys
import time
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

import ocr_extract
import ocr_report

BASE = Path(__file__).parent
DEMO_DIR = BASE / "demo"
OUT_DIR = DEMO_DIR / "out"
REPORT = DEMO_DIR / "시연_결과.html"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEMO_TIMEOUT = 30
GEMINI_BACKUP_MODELS = ["gemini-3.5-flash-lite"]

FIELD_ROWS = ocr_report.FIELD_ROWS


def step(title):
    print(f"\n  ── {title} " + "─" * max(0, 50 - display_width(title)))


@contextlib.contextmanager
def quiet():
    """PaddlePaddle(C++)가 콘솔에 직접 찍는 로그를 숨긴다.

    Windows 콘솔에서는 fd 를 NUL 로 돌린 동안 sys.stdout/stderr 에 쓰면 OSError 가
    나므로, 파이썬 쪽 출력도 같은 동안 devnull 파일로 바꿔 둔다.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_fds = os.dup(1), os.dup(2)
    saved_streams = sys.stdout, sys.stderr
    devnull = os.open(os.devnull, os.O_WRONLY)
    sink = open(os.devnull, "w", encoding="utf-8")
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    sys.stdout = sys.stderr = sink
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved_streams
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        sink.close()
        for fd in (*saved_fds, devnull):
            os.close(fd)


def display_width(text):
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def load_ocr():
    started = time.time()
    with quiet():
        import ocr_test
        engines = {name: ocr_test.PaddleOCR(**ocr_test.COMMON, **kwargs)
                   for name, kwargs in ocr_test.MODELS.items()}
    print(f"  PaddleOCR 모델 로딩 완료 ({time.time() - started:.1f}초)")
    return ocr_test, engines


def run_ocr(ocr_test, engines, image):
    with quiet():
        results = {name: ocr_test.run_model(engine, image) for name, engine in engines.items()}
    chosen = max(results, key=lambda name: results[name]["avg_score"])
    best = results[chosen]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for obj in best["objects"]:
        try:
            obj.save_to_img(str(OUT_DIR))
        except Exception as error:
            print(f"  [경고] 인식 영역 이미지 저장 실패: {error}")

    return {
        "image": image.name,
        "lines": best["lines"],
        "avg_score": best["avg_score"],
        "chosen_model": chosen,
        "language": ocr_test.detect_language(" ".join(best["texts"])),
    }


def run_llm(engine, image, ocr):
    picture = ocr_extract.encode_image(image) if ocr_extract.SEND_IMAGE else None
    prompt = ocr_extract.ocr_prompt(ocr)
    models = [engine.model]
    if engine.name == "gemini":
        models += [m for m in GEMINI_BACKUP_MODELS if m != engine.model]

    primary = engine.model
    try:
        for index, model in enumerate(models):
            engine.model = model
            try:
                fields, model, _, _ = ocr_extract.extract_with_retry(engine, picture, prompt)
                break
            except ocr_extract.ExtractError as error:
                if index == len(models) - 1:
                    raise
                print(f"    {model} 응답 실패 → {models[index + 1]} 로 재시도")
    finally:
        engine.model = primary
    return {
        "image_path": f"demo/{image.name}",
        "ocr_raw_text": "\n".join(ocr["lines"]),
        "language": ocr["language"],
        "confidence_score": ocr["avg_score"],
        "has_handwritten_note": fields["has_handwritten_note"],
        "domain": ocr_extract.domain_of(fields),
        "warnings": ocr_extract.check(fields),
        "engine": engine.name,
        "model": model,
        "llm_structured_json": fields,
    }


def load_previous(image):
    """API 실패 시 대신 보여줄, 이전 실행에서 demo/out 에 저장된 결과."""
    path = OUT_DIR / f"{image.stem}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def print_fields(card):
    fields = card["llm_structured_json"]
    for label, path in FIELD_ROWS:
        value = ocr_report.pick(fields, path)
        if isinstance(value, list):
            value = " › ".join(value) if path == "departments" else ", ".join(value)
        if value:
            print(f"    {label:<8}{value}")
    if card.get("domain"):
        print(f"    {'도메인':<8}{card['domain']}")
    for warning in card["warnings"]:
        print(f"    !  {warning}")


def find_visual(image):
    if OUT_DIR.exists():
        matches = sorted(OUT_DIR.glob(f"{image.stem}_ocr_res_img.*"))
        if matches:
            return matches[0]
    return image


def render_html(entries):
    esc = ocr_report.esc
    parts = [
        '<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>명함 OCR 시연</title>',
        f'<style>{ocr_report.STYLE}</style></head><body>',
        '<header><h1>명함 OCR → 구조화 시연</h1>',
        '<div class="meta">PaddleOCR + LLM 구조화 · '
        f'{time.strftime("%Y-%m-%d %H:%M")}</div></header><main>',
    ]

    for entry in entries:
        card, ocr = entry["card"], entry["ocr"]
        source = "저장된 결과" if entry["cached"] else f'{card["engine"]} · {card["model"]}'
        parts.append(
            f'<div class="card"><div class="card-head">'
            f'<div class="name mono">{esc(ocr["image"])}</div>'
            f'<span class="badge lang">{esc(ocr_report.LANG_LABEL.get(card["language"], "?"))}</span>'
            f'<span class="badge model">{esc(source)}</span>'
            f'<span class="badge mono">OCR 신뢰도 {card["confidence_score"]:.3f}</span></div>'
            f'<div class="card-body"><div class="pane">'
            f'<img src="{ocr_report.encode_image(entry["visual"])}" alt="">'
            f'<div class="block"><div class="title">① OCR 인식 결과</div><div class="lines mono">')
        parts += [f'<div>{esc(line)}</div>' for line in ocr["lines"]]
        parts.append('</div></div></div><div class="pane">')
        parts.append(ocr_report.render_fields(card).replace("LLM 구조화 결과", "② LLM 구조화 결과"))
        parts.append('<div class="block"><div class="title">③ 저장될 JSON (llm_structured_json)</div>'
                     '<div class="lines mono" style="max-height:340px;white-space:pre-wrap">'
                     f'{esc(json.dumps(card["llm_structured_json"], ensure_ascii=False, indent=2))}'
                     '</div></div></div></div></div>')

    parts.append('</main><div class="zoom" id="zoom"><img id="zoom-img" alt=""></div>'
                 f'<script>{ocr_report.SCRIPT}</script></body></html>')
    REPORT.write_text("".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="명함 이미지 → OCR → LLM 구조화 시연")
    parser.add_argument("images", nargs="*", help="명함 이미지 (생략하면 demo/ 폴더 전체)")
    parser.add_argument("--engine", choices=list(ocr_extract.ENGINES), default=ocr_extract.DEFAULT_ENGINE)
    parser.add_argument("--model", help="LLM 모델 이름 직접 지정")
    parser.add_argument("--offline", action="store_true", help="API를 부르지 않고 저장된 결과만 사용")
    parser.add_argument("--no-open", action="store_true", help="끝나고 브라우저를 열지 않음")
    args = parser.parse_args()

    images = [Path(p) for p in args.images] or sorted(
        p for p in DEMO_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    missing = [p for p in images if not p.exists()]
    if missing or not images:
        print(f"[오류] 이미지가 없습니다: {', '.join(map(str, missing)) or DEMO_DIR}")
        sys.exit(1)

    load_dotenv(BASE / ".env")
    ocr_extract.REQUEST_TIMEOUT = DEMO_TIMEOUT
    ocr_extract.RATE_LIMIT_RETRIES = 0

    print(f"\n  명함 {len(images)}장 시연\n")
    ocr_test, ocr_engines = load_ocr()

    llm = None
    if not args.offline:
        try:
            llm = ocr_extract.ENGINES[args.engine]()
            if args.model:
                llm.model = args.model
            print(f"  LLM: {llm.name} ({llm.model})")
        except (ImportError, ocr_extract.AuthError) as error:
            print(f"  [경고] LLM 준비 실패 → 저장된 결과 사용: {error}")

    entries = []
    for index, image in enumerate(images, start=1):
        print(f"\n{'=' * 60}\n  [{index}/{len(images)}] {image.name}")

        step("① PaddleOCR 문자 인식")
        started = time.time()
        ocr = run_ocr(ocr_test, ocr_engines, image)
        for line in ocr["lines"]:
            print(f"    {line}")
        print(f"    ({time.time() - started:.1f}초 · 신뢰도 {ocr['avg_score']:.3f})")

        step("② LLM 필드 구조화")
        started = time.time()
        card, cached = None, False
        if llm:
            try:
                card = run_llm(llm, image, ocr)
                print(f"    ({time.time() - started:.1f}초 · {card['model']})")
            except (ocr_extract.ExtractError, ocr_extract.AuthError, json.JSONDecodeError) as error:
                print(f"    [API 실패] {error}")

        if card is None:
            card = load_previous(image)
            if not card:
                print("    저장된 결과도 없어 건너뜁니다.")
                continue
            cached = True
            print("    (저장된 결과 사용)")

        print_fields(card)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        target = OUT_DIR / f"{image.stem}.json"
        target.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    → {target.relative_to(BASE)}")
        entries.append({"ocr": ocr, "card": card, "cached": cached, "visual": find_visual(image)})

    if not entries:
        sys.exit(1)

    render_html(entries)
    print(f"\n{'=' * 60}\n  ③ 결과 화면: {REPORT}\n")
    if not args.no_open:
        webbrowser.open(REPORT.resolve().as_uri())


if __name__ == "__main__":
    main()
