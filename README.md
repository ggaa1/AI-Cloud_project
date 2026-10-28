# 명함 OCR 테스트 (PaddleOCR)

명함 이미지를 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)로 인식하고, 정답지(GT)와 비교해 정확도를 측정한 뒤 HTML 리포트로 정리하는 테스트 도구입니다.

- 한국어 모델(`korean`)과 기본 모델(`default`)을 모두 돌려 평균 신뢰도가 높은 쪽을 자동 채택
- 인식된 글자에서 언어(한국어 / 일본어 / 중국어 / 영어)를 자동 분류
- 정답지 기준으로 내용 포함률 · 문자 정확도 · 줄 일치율 측정
- (선택) LLM(Gemini 또는 Claude)으로 OCR 결과를 회사명 · 이름 · 이메일 등 필드 JSON으로 구조화
- 결과를 이미지까지 포함한 단일 HTML 파일로 생성 (파일 하나만 공유하면 됨)

## 폴더 구조

```
OCR 테스트/
├── cards/            # 테스트할 명함 이미지 (.jpg .jpeg .png .bmp .webp)
├── gt/               # 정답지 (이미지와 같은 파일명의 .txt)
├── gt_fields/        # (선택) 필드 정답지 (이미지와 같은 파일명의 .json)
├── out/              # ocr_test.py 실행 결과 (자동 생성)
│   ├── images/       # 인식 영역이 표시된 이미지
│   └── structured/   # ocr_extract.py 실행 결과 (필드 JSON)
├── ocr_test.py       # OCR 실행 + 정확도 측정
├── ocr_extract.py    # (선택) LLM으로 필드 구조화
├── ocr_report.py     # HTML 리포트 생성
├── demo.py           # 시연용: 이미지 몇 장을 OCR → LLM 구조화 → HTML 까지 한 번에
├── demo/             # 시연용 샘플 명함 (가상 데이터) · 결과(out/)
├── .env.example      # API 키 설정 양식 (.env 로 복사해서 사용)
└── requirements.txt
```

## 설치

Python **3.12** 기준으로 테스트했습니다. 3.14는 PaddlePaddle이 지원하지 않을 수 있으니 3.12로 venv를 만드세요. PC에 여러 버전이 있으면 `py --list`로 확인할 수 있습니다.

### Windows (PowerShell)

```powershell
cd "OCR 테스트"
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> `Activate.ps1` 실행이 막히면 먼저 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 를 실행하세요.

### macOS / Linux

```bash
cd "OCR 테스트"
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> CPU 버전 PaddlePaddle이 설치됩니다. GPU를 쓰려면 [PaddlePaddle 설치 안내](https://www.paddlepaddle.org.cn/en/install/quick)에 따라 `paddlepaddle-gpu`를 따로 설치하세요.

## 실행

```bash
# 1. OCR 실행 + 정확도 측정 (콘솔에 결과 출력, out/ 에 저장)
python ocr_test.py

# 2. (선택) LLM 필드 구조화 → out/structured/  — API 키 필요, 아래 참고
python ocr_extract.py

# 3. HTML 리포트 생성 → OCR_테스트_결과.html
python ocr_report.py
```

처음 실행할 때 PaddleOCR 모델 파일을 자동으로 내려받으므로 인터넷 연결이 필요하고 몇 분 걸릴 수 있습니다. (모델은 사용자 홈의 `.paddlex` 폴더에 저장되어 다음부터는 바로 실행됩니다.)

생성된 `OCR_테스트_결과.html`은 이미지가 내장되어 있어 브라우저로 열거나 파일 그대로 다른 사람에게 전달할 수 있습니다.

## 내 명함으로 테스트하기

1. `cards/` 폴더에 명함 이미지를 넣습니다.
2. (선택) `gt/` 폴더에 **이미지와 같은 이름**의 `.txt` 정답지를 만듭니다.
   - 예: `cards/홍길동.jpg` → `gt/홍길동.txt`
   - UTF-8 인코딩, 명함에 적힌 글자를 한 줄씩 입력
   ```
   홍길동
   Hong Gil Dong
   개발팀 매니저
   서울시 강남구 테헤란로 123
   ```
   - 정답지가 없으면 인식 결과만 저장되고 정확도는 "정답지 없음"으로 표시됩니다.
3. `python ocr_test.py` → `python ocr_report.py` 순서로 실행합니다.

## LLM 필드 구조화 (ocr_extract.py)

`out/*.json`의 OCR 결과와 명함 이미지를 LLM에 보내, 아래 구조의 JSON으로 받아 `out/structured/<파일명>.json`에 저장합니다. 사내 DB 스키마(company → site → department → person, card)에 맞춘 형태입니다.

```json
{
  "image_path": "cards/홍길동.jpg",
  "ocr_raw_text": "...",
  "language": "ko",
  "confidence_score": 0.97,
  "has_handwritten_note": false,
  "domain": "example.com",
  "warnings": [],
  "llm_structured_json": {
    "company": { "name", "brand_name", "aliases", "registration_no", "homepage_url" },
    "site": { "name", "site_type", "address", "tel", "fax" },
    "departments": ["본부", "팀"],
    "person": { "name", "name_reading", "aliases", "title", "secondary_title", "email", "mobile", "tel" },
    "has_handwritten_note": false,
    "extra": { "phones", "emails", "addresses", "unmapped" }
  }
}
```

- 명함에 없는 값은 `null`(또는 빈 배열)로 두고 추측하지 않도록 지시합니다.
- `domain`은 LLM이 아니라 이메일/홈페이지에서 코드로 뽑습니다.
- `confidence_score`는 PaddleOCR 평균 신뢰도이고, `warnings`에 이메일·전화번호 형식 검증 결과가 들어갑니다.
- `departments`는 위 → 아래 순서라서 `parent_department_id`, `level`을 바로 만들 수 있습니다.

### 엔진 선택

| 엔진 | 모델 | 비용 | 키 환경변수 |
|---|---|---|---|
| `gemini` (기본) | `gemini-3.6-flash` | 저렴 (무료 등급 여부·한도는 AI Studio에서 확인) | `GEMINI_API_KEY` |
| `claude` | `claude-opus-5-5` | 종량제 (크레딧 충전 필요) | `ANTHROPIC_API_KEY` |

모델 이름은 `ocr_extract.py` 상단의 `GEMINI_MODEL`, `CLAUDE_MODEL`에서 바꿀 수 있습니다.

> ⚠️ **개인정보 주의** — 명함 이미지와 OCR 텍스트가 외부 API로 전송됩니다.
> - **Gemini 무료 등급은 보낸 데이터를 Google이 제품 개선에 사용할 수 있습니다.** 실제 고객·거래처 명함은 결제를 연결한 유료 등급에서 돌리고, 무료 등급은 공개 가능한 샘플로만 테스트하세요.
> - 사내 정책을 확인한 뒤 사용하세요. 이미지 전송을 끄려면 `ocr_extract.py`의 `SEND_IMAGE = False` (이 경우 손글씨 판단은 항상 false).

### API 키 설정

프로젝트 폴더의 `.env` 파일에 키를 적어두면 `ocr_extract.py`가 자동으로 읽습니다. `.env`가 없으면 `.env.example`을 복사해서 만드세요.

```powershell
copy .env.example .env      # Windows
```
```bash
cp .env.example .env        # macOS / Linux
```

```ini
# .env
GEMINI_API_KEY=AIza...        # Gemini (기본) — https://aistudio.google.com/apikey
ANTHROPIC_API_KEY=sk-ant-...  # Claude (--engine claude) — https://platform.claude.com/ (크레딧 충전 필요)
```

- 사용할 엔진의 키만 채우면 됩니다.
- `.env`는 `.gitignore`에 들어 있어 저장소에 올라가지 않습니다. **키가 든 `.env`는 다른 사람에게 보내지 마세요.** 공유할 때는 `.env.example`만 포함합니다.
- 터미널 환경변수에 같은 이름의 키가 이미 있으면 그 값이 우선합니다.

### 실행 옵션

```bash
python ocr_extract.py --limit 2          # 앞에서 2장만 (비용·결과 먼저 확인)
python ocr_extract.py                    # 전체, Gemini (같은 엔진으로 이미 처리한 명함은 건너뜀)
python ocr_extract.py --engine claude    # Claude로 처리
python ocr_extract.py --model gemini-3.8-flash   # 모델 직접 지정
python ocr_extract.py --force            # 전체 다시 처리
```

요청 한도(429)·서버 과부하(503)·응답 없음(90초)이면 30초 기다렸다가 최대 3번 다시 시도합니다.

실행이 끝나면 사용한 토큰 수가 출력됩니다. `python ocr_report.py`를 다시 실행하면 리포트의 명함별 상세에 구조화 결과가 함께 표시됩니다.

### 필드 정확도 측정 (선택)

`gt_fields/<이미지와 같은 파일명>.json`에 정답 필드를 넣어두면 필드별 정확도를 계산합니다. 정답지에 있는 섹션(`person`, `company`, `site`)만 비교하며, 공백·대소문자는 무시하고 전화번호는 숫자만 비교합니다.

```json
{
  "company": { "name": "씽크트리" },
  "site": { "address": "[03909] 서울시 마포구 매봉산로 37, DMC산학협력연구센터 4층 404호", "tel": null, "fax": "02 6959 5968" },
  "person": { "name": "이민균", "name_reading": "Lee Min Kyoon", "title": "운영책임이사",
              "email": "mklee70@think-tree.co.kr", "mobile": "010-8905-4432", "tel": "02-309-2900" }
}
```

## 시연 (demo.py)

지정한 이미지만 OCR → LLM 구조화 → 결과 HTML 까지 한 번에 처리하고, 단계별 결과를 콘솔에 보여줍니다.

```bash
python demo.py                          # demo/ 폴더의 이미지 전부
python demo.py 내명함.jpg                # 특정 이미지
python demo.py --offline                # API 없이 저장된 결과로 시연
```

- 결과는 `demo/시연_결과.html`로 저장되고 브라우저가 자동으로 열립니다 (`--no-open`으로 끔). 명함별 JSON은 `demo/out/<파일명>.json`에 저장됩니다.
- API가 30초 안에 답하지 않거나 과부하(503)면 `gemini-3.5-flash-lite`로 한 번 더 시도하고, 그래도 실패하면 `demo/out/`에 남아 있는 이전 실행 결과를 보여줍니다.
- 시연 전에 한 번 성공시켜 두면 네트워크가 안 돼도 (`--offline`) 시연할 수 있습니다.

## 결과 지표

| 지표 | 의미 |
|---|---|
| **내용 포함률** | 정답지의 각 줄이 OCR 결과 어딘가에 들어 있는 비율 (줄 단위 문자 오류 20% 이내 허용). 줄 순서·줄바꿈이 달라도 내용만 맞으면 인정 |
| **문자 정확도** | 공백을 제거한 전체 텍스트의 편집거리 기반 `1 - CER` |
| **줄 일치율** | 정답지의 각 줄과 거의 같은 줄(오류 20% 이내)이 OCR 결과에 따로 존재하는 비율. 줄 구분까지 맞아야 인정 |
| **신뢰도** | PaddleOCR가 반환한 인식 점수의 평균 |

## 출력 파일

- `out/<파일명>.txt` — 줄 단위로 정리된 인식 결과
- `out/<파일명>.json` — 채택 모델, 신뢰도, 두 모델의 원본 결과(텍스트·점수·좌표)
- `out/images/` — 인식 영역이 표시된 시각화 이미지
- `out/structured/<파일명>.json` — LLM 필드 구조화 결과
- `OCR_테스트_결과.html` — 종합 리포트

## 설정 바꾸기

`ocr_test.py` 상단의 상수로 조정할 수 있습니다.

- `MODELS` — 비교할 PaddleOCR 모델 (`lang` 값 추가 가능, 예: `"japan": dict(lang="japan")`)
- `LINE_TOLERANCE` — 같은 줄로 묶는 기준 (글자 높이 대비 비율)
- `MATCH_THRESHOLD` — 줄 일치로 인정하는 최대 오류율 (기본 0.2)
- `LANG_OVERRIDE` — 언어 자동 분류가 틀린 이미지를 수동 지정, 예: `{"홍길동": "ko"}`

## 문제 해결

- **`No module named ...` / `DLL load failed`** — venv를 다른 Python 버전으로 다시 만든 경우입니다. `venv` 폴더를 지우고 위 설치 과정(`py -3.12 -m venv venv` → `pip install -r requirements.txt`)을 처음부터 다시 하세요.

- **`API 키가 없습니다` / `잘못되었습니다`** — `.env` 파일이 프로젝트 폴더에 있는지, 엔진에 맞는 키(`GEMINI_API_KEY` 또는 `ANTHROPIC_API_KEY`)가 따옴표나 공백 없이 들어 있는지 확인하세요.

- **콘솔 한글이 깨짐** — `chcp 65001` 실행 후 다시 실행하거나, 환경변수 `PYTHONUTF8=1` 을 설정하세요. (결과 파일은 항상 UTF-8로 저장됩니다.)
- **모델 다운로드 실패** — 사내망·프록시 환경에서는 외부 접속이 막혀 있을 수 있습니다. 인터넷이 되는 환경에서 한 번 실행해 홈 폴더의 `.paddlex` 폴더를 복사해 오면 됩니다.
