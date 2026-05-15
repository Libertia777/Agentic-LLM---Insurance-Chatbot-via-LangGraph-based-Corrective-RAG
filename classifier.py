from __future__ import annotations

import os
import re
from typing import Dict, Optional, Literal

from openai import OpenAI
from google.colab import userdata
from pydantic import BaseModel

api_key = ""
os.environ["OPENAI_API_KEY"] = api_key

LABELS_KO: Dict[str, str] = {
    # Conversation/control
    "GREET": "인사",
    "GOODBYE": "대화 종료",
    "FEEDBACK_POS": "긍정 피드백/감사",
    "FEEDBACK_NEG": "부정 피드백/불만",
    "HUMAN_HANDOFF": "상담원 연결 요청",

    # Pre-purchase
    "PRODUCT_OVERVIEW": "상품 개요",
    "PLAN_OPTIONS": "플랜/특약 옵션",
    "COVERAGE_BENEFITS": "보장 내용/급부",
    "EXCLUSIONS_LIMITATIONS": "면책/제한 사항",
    "WAITING_PERIOD": "대기기간/면책기간",
    "PREEXISTING_CONDITIONS": "기존 질병/병력",
    "ELIGIBILITY_UNDERWRITING": "가입 가능 여부/심사",
    "PREMIUM_STABILITY_RENEWAL": "보험료 고정/갱신",
    "PRICING_GENERAL": "일반 보험료/가격",
    "QUOTE_PERSONALIZED": "개인 맞춤 견적",
    "PAYMENT_BILLING": "납입/결제/청구",
    "COORDINATION_OTHER_INSURANCE": "타 보험과의 연계/중복",
    "APPLICATION_HOWTO": "가입/청약 방법",

    # Claims / servicing
    "CLAIM_FILING": "보험금 청구 방법",
    "CLAIM_DOCUMENTS": "청구 서류",
    "CLAIM_STATUS_TIMELINE": "청구 진행/처리 기간",
    "POLICY_CHANGES_CANCEL_RENEW": "계약 변경/해지/갱신",
    "COMPLAINT_DISPUTE": "민원/이의제기",

    # Safety/compliance routing
    "PRIVACY_PII": "개인정보/민감정보",
    "MEDICAL_DIAGNOSIS_ADVICE": "의료 상담/진단/치료 조언",
    "OUT_OF_SCOPE": "기타/범위 외",
}

LabelKey = Literal[
    "GREET",
    "GOODBYE",
    "FEEDBACK_POS",
    "FEEDBACK_NEG",
    "HUMAN_HANDOFF",
    "PRODUCT_OVERVIEW",
    "PLAN_OPTIONS",
    "COVERAGE_BENEFITS",
    "EXCLUSIONS_LIMITATIONS",
    "WAITING_PERIOD",
    "PREEXISTING_CONDITIONS",
    "ELIGIBILITY_UNDERWRITING",
    "PREMIUM_STABILITY_RENEWAL",
    "PRICING_GENERAL",
    "QUOTE_PERSONALIZED",
    "PAYMENT_BILLING",
    "COORDINATION_OTHER_INSURANCE",
    "APPLICATION_HOWTO",
    "CLAIM_FILING",
    "CLAIM_DOCUMENTS",
    "CLAIM_STATUS_TIMELINE",
    "POLICY_CHANGES_CANCEL_RENEW",
    "COMPLAINT_DISPUTE",
    "PRIVACY_PII",
    "MEDICAL_DIAGNOSIS_ADVICE",
    "OUT_OF_SCOPE",
]

_RE_SSN = re.compile(r"\b\d{{6}}-\d{{7}}\b")  # 주민등록번호
_RE_PHONE = re.compile(r"\b01[016789]-?\d{3,4}-?\d{4}\b") #전화번호
_RE_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")  # 카드번호

class QueryClassification(BaseModel):
    label: LabelKey
    reason_ko: str  # 짧게 (1~2문장) 근거


def _build_system_prompt(labels_ko: Dict[str, str]) -> str:
    # 라벨 정의를 "핵심 기준" 위주로만 짧게 적어, 모델이 헷갈리지 않게 합니다.
    # (너무 길면 오히려 분류 품질이 떨어지는 경우가 많습니다.)
    label_lines = "\n".join([f"- {k}: {v}" for k, v in labels_ko.items()])

    return f"""너는 한국어 사용자 질의를 아래 라벨 중 '정확히 하나'로 분류하는 라우터다.
            반드시 스키마에 맞춰 JSON으로만 응답한다.

            [라벨 목록]
            {label_lines}

            [분류 규칙(중요)]
            - 대화 행위(인사/종료/피드백/상담원연결)는 보험 내용보다 우선한다.
            - 사용자가 주민번호/전화번호/주소/계좌/카드번호 등 개인정보를 제공하거나 요청하면: PRIVACY_PII
            - 진단/치료/약 복용/검사 해석 등 의료 조언 요청이면: MEDICAL_DIAGNOSIS_ADVICE
            - 보험료가 '고정인지/갱신형인지/갱신 시 인상'처럼 구조를 묻는다면: PREMIUM_STABILITY_RENEWAL
            - 그냥 "얼마에요/보험료/가격"이면: PRICING_GENERAL
            - 개인 정보(나이/성별/직업/병력 등)를 넣어 내 견적을 내달라: QUOTE_PERSONALIZED
            - 보장/급부/지급 조건: COVERAGE_BENEFITS
            - 면책/제한/불지급: EXCLUSIONS_LIMITATIONS
            - 대기기간/면책기간: WAITING_PERIOD
            - 기존 질병/병력으로 가입 가능?: PREEXISTING_CONDITIONS 또는 ELIGIBILITY_UNDERWRITING(심사/인수/가입가능여부가 중심이면)
            - 청구 방법: CLAIM_FILING / 청구서류: CLAIM_DOCUMENTS / 처리기간/진행상태: CLAIM_STATUS_TIMELINE
            - 계약 변경/해지/갱신/철회: POLICY_CHANGES_CANCEL_RENEW
            - 민원/이의제기/분쟁: COMPLAINT_DISPUTE
            - 위 범주로 보기 어렵거나 보험과 무관하면: OUT_OF_SCOPE

            [예시]
            - "안녕하세요" -> GREET
            - "상담원 연결해줘" -> HUMAN_HANDOFF
            - "보험료 갱신형이야? 오르나?" -> PREMIUM_STABILITY_RENEWAL
            - "월 보험료 얼마야?" -> PRICING_GENERAL
            - "97년생 남자인데 암보험 견적" -> QUOTE_PERSONALIZED
            - "암 진단받았는데 치료 어떻게?" -> MEDICAL_DIAGNOSIS_ADVICE
            """

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def _heuristic_label(query: str) -> Optional[LabelKey]:
    q = _normalize(query).lower()

    if _RE_SSN.search(q) or _RE_PHONE.search(q):
        return "PRIVACY_PII"
    if any(k in q for k in ["주민등록", "주민번호", "계좌", "카드번호",
                            "신용카드", "비밀번호", "otp", "주소",
                            "전화번호", "메일주소", "이메일"]):
        return "PRIVACY_PII"
    if "카드" in q and _RE_CARD.search(q):
        return "PRIVACY_PII"

    # 인사/종료/피드백/상담원 연결
    if any(k in q for k in ["상담원", "상담사", "직원", "사람 상담",
                            "전화 연결", "콜센터", "연결해줘", "연결해 주세요"]):
        return "HUMAN_HANDOFF"
    if any(k in q for k in ["안녕", "안녕하세요", "안녕!", "안녕?",
                            "반가워", "좋은 아침", "좋은저녁", "hi", "hello"]):
        return "GREET"
    if any(k in q for k in ["대화 종료", "종료", "종료할게", "끝낼게",
                            "그만", "bye", "잘가", "안녕히", "수고", "나갈게"]):
        return "GOODBYE"

    # -- Test Result: especially for positive feedback, it detects some words
    # -- that should be classified to other category ex: MEDICAL_DIAGNOSIS_ADVICE
    # --> 어떤 상품이 좋을까요? : classified as FEEDBACK_POS but should be classified as MEDICAL_DIAGNOSIS_ADVICE

    # if any(k in q for k in ["감사", "고마워", "덕분", "도움됐", "좋아요", "최고", "고맙습니다"]):
    #     return "FEEDBACK_POS"

    # if any(k in q for k in ["별로", "실망", "불만", "화나", "짜증", "틀렸", "엉망", "최악", "왜이래"]):
    #     return "FEEDBACK_NEG"

    return None

class LLMQueryClassifier:
    """
    - Structured Outputs(Pydantic)로 label을 강제하므로 후처리가 단순합니다.
    - store=False로 저장을 끌 수 있습니다(원하면 바꾸세요).
    """
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.system_prompt = _build_system_prompt(LABELS_KO)

    def classify(self, query: str) -> Dict[str, str]:
        query = _normalize(query)
        if not query:
            return {"label": "OUT_OF_SCOPE", "label_ko": LABELS_KO["OUT_OF_SCOPE"], "reason_ko": "빈 입력입니다."}

        # (선택) 로컬 규칙 우선
        h = _heuristic_label(query)
        if h is not None:
            return {"label": h, "label_ko": LABELS_KO[h], "reason_ko": "로컬 규칙에 의해 분류되었습니다."}

        # LLM 호출
        try:
            resp = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ],
                text_format=QueryClassification,
                # 안전/개인정보 측면에서 원하면 저장 비활성화
                store=False,
                temperature=0,
            )
            parsed: QueryClassification = resp.output_parsed
            return {
                "label": parsed.label,
                "label_ko": LABELS_KO[parsed.label],
                "reason_ko": parsed.reason_ko.strip(),
            }
        except Exception as e:
            # 장애 시: 최소 동작 보장
            return {
                "label": "OUT_OF_SCOPE",
                "label_ko": LABELS_KO["OUT_OF_SCOPE"],
                "reason_ko": f"분류 중 오류가 발생해 OUT_OF_SCOPE로 처리했습니다: {type(e).__name__}",
            }

