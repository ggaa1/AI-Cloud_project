import base64
import html
import io
import json
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
GT_DIR = BASE / "gt"
OUT_DIR = BASE / "out"
IMAGE_DIR = OUT_DIR / "images"
CARDS_DIR = BASE / "cards"
STRUCTURED_DIR = OUT_DIR / "structured"
REPORT = BASE / "OCR_테스트_결과.html"

MATCH_THRESHOLD = 0.2
MAX_IMAGE_WIDTH = 1100
JPEG_QUALITY = 78

LANG_LABEL = {"ko": "한국어", "ja": "일본어", "zh": "중국어", "en": "영어"}
LANG_ORDER = ["ko", "ja", "zh", "en"]


def squeeze(text):
    return "".join(text.split())


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
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(truth, pred):
    if not truth:
        return 0.0 if not pred else 1.0
    return levenshtein(truth, pred) / len(truth)


def infix_distance(needle, haystack):
    if not needle:
        return 0
    if not haystack:
        return len(needle)

    previous = [0] * (len(haystack) + 1)
    for ca in needle:
        current = [previous[0] + 1]
        for j, cb in enumerate(haystack, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return min(previous)


def evaluate(gt_lines, ocr_lines):
    targets = [line for line in gt_lines if squeeze(line)]
    if not targets:
        return None

    haystack = squeeze(" ".join(ocr_lines))
    details = []
    for gt_line in targets:
        needle = squeeze(gt_line)
        score = infix_distance(needle, haystack) / len(needle)
        details.append({"ok": score <= MATCH_THRESHOLD, "line": gt_line, "score": score})

    matched = 0
    for gt_line in targets:
        best = min((cer(squeeze(gt_line), squeeze(l)) for l in ocr_lines), default=1.0)
        matched += best <= MATCH_THRESHOLD

    return {
        "coverage": sum(1 for d in details if d["ok"]) / len(targets),
        "accuracy": 1 - cer(squeeze("\n".join(targets)), haystack),
        "recall": matched / len(targets),
        "details": details,
    }


def encode_image(path):
    raw = path.read_bytes()
    try:
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert("RGB")
            if image.width > MAX_IMAGE_WIDTH:
                height = int(image.height * MAX_IMAGE_WIDTH / image.width)
                image = image.resize((MAX_IMAGE_WIDTH, height), Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=JPEG_QUALITY)
            raw = buffer.getvalue()
        mime = "image/jpeg"
    except Exception:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def find_visualization(stem):
    for directory in (IMAGE_DIR, CARDS_DIR):
        if not directory.exists():
            continue
        matches = sorted(p for p in directory.iterdir()
                         if p.is_file() and stem in p.stem
                         and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if matches:
            return matches[0]
    return None


def load_structured(stem):
    path = STRUCTURED_DIR / f"{stem}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_cards():
    cards = []
    for path in sorted(OUT_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        stem = path.stem

        gt_file = GT_DIR / f"{stem}.txt"
        gt_lines = []
        if gt_file.exists():
            gt_lines = [l for l in gt_file.read_text(encoding="utf-8").splitlines()
                        if l.strip()]

        lines = data.get("lines", [])
        cards.append({
            "stem": stem,
            "image": data.get("image", stem),
            "language": data.get("language", "en"),
            "model": data.get("chosen_model", "-"),
            "confidence": data.get("avg_score", 0.0),
            "lines": lines,
            "gt_lines": gt_lines,
            "report": evaluate(gt_lines, lines),
            "visual": find_visualization(stem),
            "structured": load_structured(stem),
        })
    return cards


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def pct(value, digits=1):
    return f"{value * 100:.{digits}f}%" if value is not None else "–"


def bar(value, tone):
    if value is None:
        return '<div class="bar"><span style="width:0"></span></div>'
    return (f'<div class="bar"><span class="{tone}" '
            f'style="width:{value * 100:.1f}%"></span></div>')


def tone_for(value):
    if value is None:
        return "flat"
    if value >= 0.9:
        return "good"
    if value >= 0.75:
        return "mid"
    return "low"


def esc(text):
    return html.escape(str(text))


STYLE = """
:root{
  --bg:#0E1013; --panel:#16181D; --panel-2:#1C1F26; --line:#262A32;
  --text:#E8EAED; --muted:#8A9099; --dim:#5C636E;
  --teal:#5FD3C4; --amber:#E8A24A; --red:#E0685F; --blue:#6EA8FE;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font-family:'Pretendard','Apple SD Gothic Neo','Malgun Gothic',system-ui,sans-serif;
  font-size:15px;line-height:1.6}
.mono{font-family:'SFMono-Regular',Consolas,'D2Coding',monospace}
header{position:sticky;top:0;z-index:10;background:var(--panel);
  border-bottom:1px solid var(--line);padding:18px 28px;
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}
header h1{margin:0;font-size:19px;letter-spacing:-.2px}
header .meta{color:var(--dim);font-size:13px}
header .meta b{color:var(--muted);font-weight:500}
main{max-width:1180px;margin:0 auto;padding:28px}
section{margin-bottom:44px}
h2{font-size:13px;letter-spacing:1.4px;text-transform:uppercase;color:var(--dim);
  margin:0 0 16px;font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px}
.kpi .label{font-size:12px;color:var(--dim);letter-spacing:.6px;text-transform:uppercase}
.kpi .value{font-size:34px;font-weight:600;margin-top:6px;letter-spacing:-1px}
.kpi .note{font-size:12px;color:var(--muted);margin-top:4px}
.v-teal{color:var(--teal)} .v-amber{color:var(--amber)} .v-text{color:var(--text)}
table{width:100%;border-collapse:collapse;background:var(--panel);
  border:1px solid var(--line);border-radius:12px;overflow:hidden}
th,td{padding:12px 16px;text-align:left;border-bottom:1px solid var(--line);font-size:14px}
th{background:var(--panel-2);color:var(--muted);font-size:12px;
  letter-spacing:.8px;text-transform:uppercase;font-weight:600}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right}
tr.total td{background:var(--panel-2);font-weight:600}
.bar{height:5px;background:#23262D;border-radius:3px;overflow:hidden;margin-top:7px}
.bar span{display:block;height:100%;border-radius:3px}
.bar .good{background:var(--teal)} .bar .mid{background:var(--amber)}
.bar .low{background:var(--red)} .bar .flat{background:var(--dim)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  margin-bottom:18px;overflow:hidden}
.card-head{display:flex;align-items:center;gap:12px;padding:14px 18px;
  border-bottom:1px solid var(--line);flex-wrap:wrap}
.card-head .name{font-size:14px;color:var(--text);flex:1;min-width:220px;
  word-break:break-all}
.badge{font-size:11px;padding:3px 9px;border-radius:20px;border:1px solid var(--line);
  color:var(--muted);background:var(--panel-2);white-space:nowrap}
.badge.lang{border-color:#2F4A52;color:var(--teal)}
.badge.model{border-color:#4A3C28;color:var(--amber)}
.card-body{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:0}
.pane{padding:18px}
.pane + .pane{border-left:1px solid var(--line)}
.pane img{width:100%;border-radius:8px;display:block;background:#000;cursor:zoom-in}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:16px}
.metric .label{font-size:11px;color:var(--dim);letter-spacing:.6px;text-transform:uppercase}
.metric .value{font-size:22px;font-weight:600;margin-top:2px}
.block{margin-top:14px}
.block .title{font-size:11px;color:var(--dim);letter-spacing:.8px;
  text-transform:uppercase;margin-bottom:7px}
.lines{background:#101216;border:1px solid var(--line);border-radius:8px;
  padding:12px 14px;font-size:13px;max-height:260px;overflow:auto}
.lines div{padding:2px 0;color:#C9CED6;word-break:break-all}
.miss{border:1px solid #4A2B28;background:#1C1416;border-radius:8px;
  padding:12px 14px;margin-top:12px}
.miss .title{color:var(--red)}
.miss div{font-size:13px;padding:3px 0;color:#D8B8B4;word-break:break-all}
.miss .score{color:var(--dim);font-size:11px;margin-left:6px}
.ok{border:1px solid #27423C;background:#121A18;border-radius:8px;
  padding:11px 14px;margin-top:12px;color:var(--teal);font-size:13px}
.note{color:var(--muted);font-size:13px;margin:-6px 0 16px}
table.fields{border-radius:8px;background:#101216}
table.fields th{width:92px;padding:7px 12px;font-size:11px;text-transform:none;letter-spacing:0}
table.fields td{padding:7px 12px;font-size:13px;word-break:break-all}
footer{color:var(--dim);font-size:12px;text-align:center;padding:30px 0 50px}
.zoom{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;
  align-items:center;justify-content:center;z-index:50;padding:24px;cursor:zoom-out}
.zoom.on{display:flex}
.zoom img{max-width:100%;max-height:100%;border-radius:8px}
@media (max-width:900px){
  .card-body{grid-template-columns:1fr}
  .pane + .pane{border-left:none;border-top:1px solid var(--line)}
  main{padding:18px}
}
"""

SCRIPT = """
document.addEventListener('click', function(e){
  var zoom = document.getElementById('zoom');
  if (e.target.tagName === 'IMG' && e.target.closest('.pane')) {
    document.getElementById('zoom-img').src = e.target.src;
    zoom.classList.add('on');
  } else if (zoom.classList.contains('on')) {
    zoom.classList.remove('on');
  }
});
document.addEventListener('keydown', function(e){
  if (e.key === 'Escape') document.getElementById('zoom').classList.remove('on');
});
"""


def render_kpis(cards):
    measured = [c for c in cards if c["report"]]
    coverage = mean([c["report"]["coverage"] for c in measured])
    accuracy = mean([c["report"]["accuracy"] for c in measured])
    recall = mean([c["report"]["recall"] for c in measured])

    items = [
        ("내용 포함률", pct(coverage), "v-teal", "정답 줄이 결과에 살아있는 비율"),
        ("문자 정확도", pct(accuracy), "v-amber", "CER 기반 글자 단위"),
        ("줄 일치율", pct(recall), "v-text", "줄 구분까지 동일"),
        ("측정 명함", f"{len(measured)}<span style='font-size:18px;color:#5C636E'>"
                      f" / {len(cards)}</span>", "v-text", "정답지가 있는 장수"),
    ]

    html_parts = ['<div class="kpis">']
    for label, value, tone, note in items:
        html_parts.append(
            f'<div class="kpi"><div class="label">{esc(label)}</div>'
            f'<div class="value mono {tone}">{value}</div>'
            f'<div class="note">{esc(note)}</div></div>')
    html_parts.append('</div>')
    return "".join(html_parts)


def render_language_table(cards):
    rows = []
    for language in LANG_ORDER:
        group = [c for c in cards if c["language"] == language]
        if not group:
            continue
        measured = [c for c in group if c["report"]]
        rows.append((LANG_LABEL[language], len(group), len(measured),
                     mean([c["report"]["coverage"] for c in measured]),
                     mean([c["report"]["accuracy"] for c in measured]),
                     mean([c["report"]["recall"] for c in measured])))

    measured = [c for c in cards if c["report"]]
    total = ("전체", len(cards), len(measured),
             mean([c["report"]["coverage"] for c in measured]),
             mean([c["report"]["accuracy"] for c in measured]),
             mean([c["report"]["recall"] for c in measured]))

    out = ['<table><thead><tr><th>언어</th><th class="num">장수</th>'
           '<th class="num">측정</th><th class="num">내용 포함률</th>'
           '<th class="num">문자 정확도</th><th class="num">줄 일치율</th>'
           '</tr></thead><tbody>']
    for label, count, measured_count, coverage, accuracy, recall in rows + [total]:
        css = ' class="total"' if label == "전체" else ""
        out.append(
            f'<tr{css}><td>{esc(label)}</td>'
            f'<td class="num mono">{count}</td>'
            f'<td class="num mono">{measured_count}</td>'
            f'<td class="num mono">{pct(coverage)}{bar(coverage, tone_for(coverage))}</td>'
            f'<td class="num mono">{pct(accuracy)}</td>'
            f'<td class="num mono">{pct(recall)}</td></tr>')
    out.append('</tbody></table>')
    return "".join(out)


FIELD_ROWS = [
    ("이름", "person.name"), ("이름 읽기", "person.name_reading"),
    ("직함", "person.title"), ("보조 직함", "person.secondary_title"),
    ("부서", "departments"), ("회사", "company.name"), ("회사 별칭", "company.aliases"),
    ("이메일", "person.email"), ("휴대폰", "person.mobile"), ("전화", "person.tel"),
    ("대표 전화", "site.tel"), ("팩스", "site.fax"), ("주소", "site.address"),
    ("홈페이지", "company.homepage_url"),
]


def pick(data, path):
    for key in path.split("."):
        data = data.get(key) if isinstance(data, dict) else None
    return data


def render_fields(structured):
    fields = structured["llm_structured_json"]
    out = ['<div class="block"><div class="title">LLM 구조화 결과</div><table class="fields">']
    for label, path in FIELD_ROWS:
        value = pick(fields, path)
        if isinstance(value, list):
            value = " › ".join(value) if path == "departments" else ", ".join(value)
        if value:
            out.append(f'<tr><th>{esc(label)}</th><td>{esc(value)}</td></tr>')
    out.append('</table></div>')
    if structured.get("warnings"):
        out.append('<div class="miss"><div class="title">검증 경고</div>')
        out += [f'<div>{esc(w)}</div>' for w in structured["warnings"]]
        out.append('</div>')
    return "".join(out)


def render_card(card):
    report = card["report"]

    head = [f'<div class="card-head"><div class="name mono">{esc(card["image"])}</div>',
            f'<span class="badge lang">{esc(LANG_LABEL.get(card["language"], "?"))}</span>',
            f'<span class="badge model">{esc(card["model"])}</span>',
            f'<span class="badge mono">신뢰도 {card["confidence"]:.3f}</span></div>']

    left = ['<div class="pane">']
    if card["visual"]:
        left.append(f'<img src="{encode_image(card["visual"])}" alt="">')
    else:
        left.append('<div class="note">인식 결과 이미지 없음</div>')
    left.append('</div>')

    right = ['<div class="pane">']
    if report:
        metrics = [("포함률", report["coverage"], "v-teal"),
                   ("정확도", report["accuracy"], "v-amber"),
                   ("줄 일치", report["recall"], "v-text")]
        right.append('<div class="metrics">')
        for label, value, tone in metrics:
            right.append(
                f'<div class="metric"><div class="label">{esc(label)}</div>'
                f'<div class="value mono {tone}">{pct(value, 0)}</div>'
                f'{bar(value, tone_for(value))}</div>')
        right.append('</div>')
    else:
        right.append('<div class="note">정답지 없음 — 측정하지 않음</div>')

    right.append('<div class="block"><div class="title">OCR 인식 결과</div>'
                 '<div class="lines mono">')
    for line in card["lines"]:
        right.append(f'<div>{esc(line)}</div>')
    right.append('</div></div>')

    if card["structured"]:
        right.append(render_fields(card["structured"]))

    if report:
        misses = [d for d in report["details"] if not d["ok"]]
        if misses:
            right.append('<div class="miss"><div class="title">놓친 줄</div>')
            for item in misses:
                right.append(f'<div>{esc(item["line"])}'
                             f'<span class="score mono">오류 {item["score"]*100:.0f}%</span></div>')
            right.append('</div>')
        else:
            right.append('<div class="ok">정답지의 모든 줄이 인식됨</div>')
    right.append('</div>')

    return (f'<div class="card">{"".join(head)}'
            f'<div class="card-body">{"".join(left)}{"".join(right)}</div></div>')


def main():
    if not OUT_DIR.exists():
        print(f"[오류] {OUT_DIR} 폴더가 없습니다. ocr_test.py 를 먼저 실행하세요.")
        sys.exit(1)

    cards = load_cards()
    if not cards:
        print(f"[오류] {OUT_DIR} 안에 결과 JSON이 없습니다.")
        sys.exit(1)

    print(f"명함 {len(cards)}장 읽음. 이미지 인코딩 중...")

    parts = [
        '<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>OCR 테스트 결과</title>',
        f'<style>{STYLE}</style></head><body>',
        '<header><h1>명함 OCR 인식 테스트 결과</h1>',
        f'<div class="meta">PaddleOCR PP-OCRv5 / v6 · 샘플 <b>{len(cards)}장</b> · '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")}</div></header><main>',
        '<section><h2>종합</h2>',
        render_kpis(cards),
        '</section>',
        '<section><h2>언어별 결과</h2>',
        render_language_table(cards),
        '</section>',
    ]

    parts.append('<section><h2>명함별 상세</h2>')
    parts.append('<div class="note">이미지를 클릭하면 크게 볼 수 있습니다</div>')
    for index, card in enumerate(cards, start=1):
        print(f"  [{index}/{len(cards)}] {card['image']}")
        parts.append(render_card(card))
    parts.append('</section>')

    parts += [
        '<footer>내용 포함률 = 정답지의 각 줄이 OCR 결과 안에 존재하는 비율 '
        f'(문자 오류 {int(MATCH_THRESHOLD*100)}% 이내 허용)<br>'
        '문자 정확도 = 편집거리 기반 CER의 역수 · 줄 일치율 = 줄 구분까지 일치하는 비율'
        '</footer></main>',
        '<div class="zoom" id="zoom"><img id="zoom-img" alt=""></div>',
        f'<script>{SCRIPT}</script></body></html>',
    ]

    REPORT.write_text("".join(parts), encoding="utf-8")
    size = REPORT.stat().st_size / 1024 / 1024
    print(f"\n생성 완료: {REPORT.name}  ({size:.1f} MB)")
    print("브라우저로 열어서 확인하거나 팀원에게 파일 그대로 전달하면 됩니다.")


if __name__ == "__main__":
    main()