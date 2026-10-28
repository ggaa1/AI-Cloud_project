import argparse
import base64
import io
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).parent
CARDS_DIR = BASE / "cards"
OCR_DIR = BASE / "out"
OUT_DIR = OCR_DIR / "structured"
GT_FIELDS_DIR = BASE / "gt_fields"

DEFAULT_ENGINE = "gemini"
GEMINI_MODEL = "gemini-3.6-flash"
REQUEST_TIMEOUT = 90
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT = "medium"
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_WAIT = 30
SEND_IMAGE = True
MAX_IMAGE_SIDE = 1600
JPEG_QUALITY = 85


def nullable(kind="string"):
    return {"type": [kind, "null"]}


def string_list():
    return {"type": "array", "items": {"type": "string"}}


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


SCHEMA = obj({
    "company": obj({
        "name": nullable(),
        "brand_name": nullable(),
        "aliases": string_list(),
        "registration_no": nullable(),
        "homepage_url": nullable(),
    }),
    "site": obj({
        "name": nullable(),
        "site_type": {"type": ["string", "null"],
                      "enum": ["headquarters", "branch", "factory", "lab", "store", "other", None]},
        "address": nullable(),
        "tel": nullable(),
        "fax": nullable(),
    }),
    "departments": string_list(),
    "person": obj({
        "name": nullable(),
        "name_reading": nullable(),
        "aliases": string_list(),
        "title": nullable(),
        "secondary_title": nullable(),
        "email": nullable(),
        "mobile": nullable(),
        "tel": nullable(),
    }),
    "has_handwritten_note": {"type": "boolean"},
    "extra": obj({
        "phones": string_list(),
        "emails": string_list(),
        "addresses": string_list(),
        "unmapped": string_list(),
    }),
})

SYSTEM = """You extract structured fields from a single business card.

You receive the OCR text (lines, top to bottom) and usually the card image. Fill the JSON schema.

Rules:
- Copy values as printed on the card. Fix an OCR error only when the image clearly shows the correct characters.
- Use null (or an empty list) when the card does not show a value. Never guess or invent values.
- company.name: the company name in the card's main language. Other spellings (English name, abbreviations) go to company.aliases. brand_name only when a distinct brand/logo name differs from the legal name.
- site: the office this card belongs to. site_type is "headquarters" only when the card says so (본사, 本社, Head Office); null when unknown.
- site.tel only for a number explicitly marked as the main/representative line (대표, 代表, Main). Otherwise a landline goes to person.tel.
- person.mobile: mobile numbers (010-, M, Mobile, 携帯, 手机).
- departments: organizational units from top to bottom, e.g. ["경영지원본부", "재무팀"]. Do not include the company name or the job title.
- person.title: the job title in the card's main language. secondary_title: a second title (e.g. the English one, or a concurrent role).
- person.name_reading: the reading of the name (furigana, pinyin, or romanized/English name). Other notations go to person.aliases.
- extra.phones / extra.emails / extra.addresses: every value of that kind on the card, including the ones already placed above.
- extra.unmapped: printed lines that fit no field (slogans, certifications, SNS IDs, etc.).
- has_handwritten_note: true only if the image shows handwriting added to the card. false when no image is provided."""


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def encode_image(path):
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def ocr_prompt(ocr):
    text = "\n".join(ocr["lines"])
    return f"<ocr_text>\n{text}\n</ocr_text>"


class AuthError(Exception):
    pass


class RateLimited(Exception):
    pass


class ExtractError(Exception):
    pass


class GeminiEngine:
    name = "gemini"
    model = GEMINI_MODEL
    key_env = "GEMINI_API_KEY"

    def __init__(self):
        from google import genai
        from google.genai import errors, types
        self.errors = errors
        self.types = types
        try:
            self.client = genai.Client(http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ))
        except ValueError as error:
            raise AuthError(error)

    def extract(self, image, prompt):
        types = self.types
        contents = []
        if image:
            contents.append(types.Part.from_bytes(data=image, mime_type="image/jpeg"))
        contents.append(prompt)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    response_mime_type="application/json",
                    response_json_schema=SCHEMA,
                ),
            )
        except self.errors.ClientError as error:
            if error.code == 429:
                raise RateLimited(error)
            if error.code in (401, 403) or "API key" in str(error):
                raise AuthError(error)
            raise ExtractError(error)
        except self.errors.ServerError as error:
            if error.code == 503:
                raise RateLimited(error)
            raise ExtractError(error)
        except self.errors.APIError as error:
            raise ExtractError(error)
        except Exception as error:
            if "timeout" not in type(error).__name__.lower():
                raise
            raise RateLimited(f"{REQUEST_TIMEOUT}초 동안 응답 없음")

        if not response.text:
            reason = response.candidates[0].finish_reason if response.candidates else "응답 없음"
            raise ExtractError(f"빈 응답 ({reason})")

        usage = response.usage_metadata
        output = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        return json.loads(response.text), self.model, usage.prompt_token_count or 0, output


class ClaudeEngine:
    name = "claude"
    model = CLAUDE_MODEL
    key_env = "ANTHROPIC_API_KEY"

    def __init__(self):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()

    def extract(self, image, prompt):
        anthropic = self.anthropic
        content = []
        if image:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": base64.standard_b64encode(image).decode()},
            })
        content.append({"type": "text", "text": prompt})

        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM,
                output_config={
                    "effort": CLAUDE_EFFORT,
                    "format": {"type": "json_schema", "schema": SCHEMA},
                },
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.AuthenticationError as error:
            raise AuthError(error)
        except TypeError as error:
            if "authentication" not in str(error):
                raise
            raise AuthError("키가 설정되지 않았습니다")
        except anthropic.RateLimitError as error:
            raise RateLimited(error)
        except anthropic.APIError as error:
            raise ExtractError(error)

        if response.stop_reason == "refusal":
            raise ExtractError("모델이 요청을 거절했습니다")
        if response.stop_reason == "max_tokens":
            raise ExtractError("응답이 max_tokens 에서 잘렸습니다")

        text = next(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return json.loads(text), response.model, usage.input_tokens, usage.output_tokens


ENGINES = {engine.name: engine for engine in (GeminiEngine, ClaudeEngine)}


def extract_with_retry(engine, image, prompt):
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return engine.extract(image, prompt)
        except RateLimited as error:
            if attempt == RATE_LIMIT_RETRIES:
                raise ExtractError(f"요청 한도 초과 또는 서버 과부하: {error}")
            print(f"        요청 한도 초과 또는 서버 과부하 — {RATE_LIMIT_WAIT}초 후 재시도")
            time.sleep(RATE_LIMIT_WAIT)


def domain_of(fields):
    email = fields["person"]["email"] or next(iter(fields["extra"]["emails"]), None)
    if email and "@" in email:
        return email.split("@", 1)[1].strip().lower()

    url = fields["company"]["homepage_url"]
    if url:
        host = re.sub(r"^[a-z]+://", "", url.strip().lower()).split("/")[0]
        return host.removeprefix("www.") or None
    return None


def check(fields):
    warnings = []
    for email in [fields["person"]["email"], *fields["extra"]["emails"]]:
        if email and not EMAIL_RE.match(email):
            warnings.append(f"이메일 형식 이상: {email}")

    phones = [fields["person"]["mobile"], fields["person"]["tel"],
              fields["site"]["tel"], fields["site"]["fax"], *fields["extra"]["phones"]]
    for phone in phones:
        if phone and not 7 <= len(re.sub(r"\D", "", phone)) <= 15:
            warnings.append(f"전화번호 자릿수 이상: {phone}")

    if not fields["person"]["name"]:
        warnings.append("이름 없음")
    if not fields["company"]["name"]:
        warnings.append("회사명 없음")
    return warnings


def find_image(name):
    path = CARDS_DIR / name
    return path if path.exists() else None


# ---- 필드 정답지 비교 (gt_fields/<파일명>.json 이 있을 때만) ----

EVAL_FIELDS = [
    "company.name", "site.address", "site.tel", "site.fax",
    "person.name", "person.name_reading", "person.title",
    "person.email", "person.mobile", "person.tel",
]
PHONE_FIELDS = {"site.tel", "site.fax", "person.mobile", "person.tel"}


def pick(data, path):
    for key in path.split("."):
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def normalize(path, value):
    if value is None:
        return None
    value = str(value)
    if path in PHONE_FIELDS:
        return re.sub(r"\D", "", value) or None
    return "".join(value.split()).lower() or None


def compare(stem, fields):
    gt_file = GT_FIELDS_DIR / f"{stem}.json"
    if not gt_file.exists():
        return None
    truth = json.loads(gt_file.read_text(encoding="utf-8"))
    return {
        path: normalize(path, pick(truth, path)) == normalize(path, pick(fields, path))
        for path in EVAL_FIELDS
        if path.split(".")[0] in truth
    }


def print_field_summary(results):
    measured = [r for r in results if r]
    if not measured:
        return
    print("=" * 60)
    print(f"필드별 정확도 (정답지 {len(measured)}장)\n")
    for path in EVAL_FIELDS:
        values = [r[path] for r in measured if path in r]
        if values:
            print(f"  {path:<22}{sum(values) / len(values) * 100:6.1f}%   ({sum(values)}/{len(values)})")
    print()


def main():
    parser = argparse.ArgumentParser(description="OCR 결과를 LLM으로 필드 구조화")
    parser.add_argument("--engine", choices=list(ENGINES), default=DEFAULT_ENGINE,
                        help=f"사용할 LLM (기본 {DEFAULT_ENGINE})")
    parser.add_argument("--model", help="모델 이름 직접 지정 (예: gemini-3.8-flash)")
    parser.add_argument("--limit", type=int, help="앞에서부터 N장만 처리 (비용 확인용)")
    parser.add_argument("--force", action="store_true", help="이미 처리한 명함도 다시 처리")
    args = parser.parse_args()

    load_dotenv(BASE / ".env")

    ocr_files = sorted(OCR_DIR.glob("*.json"))
    if not ocr_files:
        print(f"[오류] {OCR_DIR} 안에 OCR 결과가 없습니다. ocr_test.py 를 먼저 실행하세요.")
        sys.exit(1)
    if args.limit:
        ocr_files = ocr_files[:args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine_class = ENGINES[args.engine]
    try:
        engine = engine_class()
        if args.model:
            engine.model = args.model
    except ImportError as error:
        print(f"[오류] {args.engine} 라이브러리가 없습니다: {error}  (pip install -r requirements.txt)")
        sys.exit(1)
    except AuthError:
        print(f"[오류] API 키가 없습니다. .env 파일의 {engine_class.key_env} 를 채우세요. (README 'API 키 설정')")
        sys.exit(1)

    print(f"명함 {len(ocr_files)}장 · {engine.name} ({engine.model}) · "
          f"이미지 전송 {'켬' if SEND_IMAGE else '끔'}\n")

    results, input_tokens, output_tokens = [], 0, 0
    for index, ocr_file in enumerate(ocr_files, start=1):
        stem = ocr_file.stem
        target = OUT_DIR / f"{stem}.json"
        print(f"[{index}/{len(ocr_files)}] {stem}")

        previous = json.loads(target.read_text(encoding="utf-8")) if target.exists() else None
        if previous and previous.get("engine") == engine.name and not args.force:
            card = previous
            print("        이미 처리됨 (다시 하려면 --force)")
        else:
            ocr = json.loads(ocr_file.read_text(encoding="utf-8"))
            image_path = find_image(ocr["image"])
            image = encode_image(image_path) if SEND_IMAGE and image_path else None
            try:
                fields, model, used_in, used_out = extract_with_retry(
                    engine, image, ocr_prompt(ocr))
            except AuthError as error:
                print(f"\n[오류] API 키가 잘못되었습니다. .env 파일의 {engine_class.key_env} 를 확인하세요: {error}")
                sys.exit(1)
            except (ExtractError, json.JSONDecodeError) as error:
                print(f"        [실패] {error}\n")
                continue

            input_tokens += used_in
            output_tokens += used_out
            card = {
                "image_path": f"cards/{ocr['image']}",
                "ocr_raw_text": "\n".join(ocr["lines"]),
                "language": ocr.get("language"),
                "confidence_score": ocr.get("avg_score"),
                "has_handwritten_note": fields["has_handwritten_note"],
                "domain": domain_of(fields),
                "warnings": check(fields),
                "engine": engine.name,
                "model": model,
                "llm_structured_json": fields,
            }
            target.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

        person = card["llm_structured_json"]["person"]
        company = card["llm_structured_json"]["company"]
        print(f"        {person['name']} · {company['name']} · {person['title']} · {person['email']}")
        for warning in card["warnings"]:
            print(f"          !  {warning}")

        report = compare(stem, card["llm_structured_json"])
        if report:
            wrong = [path for path, ok in report.items() if not ok]
            print(f"        필드 일치 {sum(report.values())}/{len(report)}"
                  + (f"   틀림: {', '.join(wrong)}" if wrong else ""))
        results.append(report)
        print()

    print_field_summary(results)
    if input_tokens:
        print(f"이번 실행 토큰: 입력 {input_tokens:,} · 출력 {output_tokens:,}")
    print(f"결과 저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
