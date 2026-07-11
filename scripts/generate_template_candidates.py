#!/usr/bin/env python3
"""Generate placeholder-only Chinese PII template candidates with a local Qwen model.

The output is quarantined candidate material.  It contains placeholders such
as ``<<PHONE_NUMBER_1>>`` but never generated PII values.  A separate admission
step validates, reviews, and versions templates before they are used to create
public training rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CORE_LABELS = (
    "PERSON_NAME",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "ADDRESS",
    "DATE_OF_BIRTH",
    "CN_RESIDENT_ID",
    "PASSPORT_NUMBER",
    "DRIVER_LICENSE_NUMBER",
    "SOCIAL_SECURITY_NUMBER",
    "BANK_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER",
    "VEHICLE_LICENSE_PLATE",
    "EMPLOYEE_ID",
    "STUDENT_ID",
    "MEDICAL_RECORD_NUMBER",
    "WECHAT_ID",
    "QQ_NUMBER",
    "ALIPAY_ACCOUNT",
    "USERNAME",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "DEVICE_ID",
    "GEO_COORDINATE",
    "SECRET",
)

RISK = {
    "CN_RESIDENT_ID": "T0",
    "PASSPORT_NUMBER": "T0",
    "DRIVER_LICENSE_NUMBER": "T0",
    "SOCIAL_SECURITY_NUMBER": "T0",
    "BANK_CARD_NUMBER": "T0",
    "BANK_ACCOUNT_NUMBER": "T0",
    "MEDICAL_RECORD_NUMBER": "T0",
    "ALIPAY_ACCOUNT": "T0",
    "GEO_COORDINATE": "T0",
    "SECRET": "T0",
    "IP_ADDRESS": "T2",
    "MAC_ADDRESS": "T2",
}

REQUESTS = (
    {
        "batch": "contact_and_identity",
        "labels": [
            "PERSON_NAME",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "ADDRESS",
        ],
        "domains": "电商、物流、客服、合同、酒店、招聘、出行",
    },
    {
        "batch": "government_documents",
        "labels": [
            "CN_RESIDENT_ID",
            "PASSPORT_NUMBER",
            "DRIVER_LICENSE_NUMBER",
            "SOCIAL_SECURITY_NUMBER",
            "PERSON_NAME",
        ],
        "domains": "政务、签证、保险、交通、社保、人事",
    },
    {
        "batch": "financial",
        "labels": [
            "BANK_CARD_NUMBER",
            "BANK_ACCOUNT_NUMBER",
            "ALIPAY_ACCOUNT",
            "PERSON_NAME",
            "PHONE_NUMBER",
        ],
        "domains": "银行、证券、保险、退款、支付、财务工单",
    },
    {
        "batch": "work_education_health",
        "labels": [
            "EMPLOYEE_ID",
            "STUDENT_ID",
            "MEDICAL_RECORD_NUMBER",
            "PERSON_NAME",
            "DATE_OF_BIRTH",
        ],
        "domains": "人力资源、校园服务、考试、门诊、住院、检验",
    },
    {
        "batch": "digital_accounts",
        "labels": [
            "WECHAT_ID",
            "QQ_NUMBER",
            "USERNAME",
            "IP_ADDRESS",
            "MAC_ADDRESS",
            "DEVICE_ID",
        ],
        "domains": "即时通信、客服、应用日志、JSON、配置、运维工单",
    },
    {
        "batch": "security_location_vehicle",
        "labels": [
            "SECRET",
            "GEO_COORDINATE",
            "VEHICLE_LICENSE_PLATE",
            "DEVICE_ID",
            "ADDRESS",
        ],
        "domains": "安全告警、车辆服务、定位、物联网、移动应用",
    },
    {
        "batch": "noisy_mixed",
        "labels": [label for label in CORE_LABELS if label != "DATE_OF_BIRTH"],
        "domains": "聊天、ASR转写、OCR、Markdown、HTML、表格、邮件、中英混排",
    },
)

SYSTEM_PROMPT = """你是中文数据工程师。
你只生成训练用的占位符模板，不生成任何真实、仿真或完整个人信息值。
输出必须是一个 JSON 数组，不能包含 Markdown、解释或思维过程。每项结构固定为：
{"id":"唯一snake_case_id","domain":"英文领域","scene":"英文场景","text":"含占位符的自然中文文本","labels":["标签"]}
占位符格式必须是 <<LABEL_N>>，例如 <<PERSON_NAME_1>> 和 <<PHONE_NUMBER_1>>。
禁止省略末尾编号，禁止使用给定标签以外的占位符。同一个占位符只出现一次。
模板应像真实客服、表单、邮件、日志、JSON、聊天或业务文本；不要出现“测试、示例、虚构、演练、占位符”等提示词。
禁止在模板中写任何完整姓名、电话号码、邮箱、证件号、银行卡号、账号、IP、MAC、坐标、密钥或其他实体值。
除普通年份、序号和 JSON 标点外不要写连续 4 位以上数字。每个模板包含 1 到 4 个占位符。
边界必须清晰，模板之间不能只是同义词替换。"""

RAW_PLACEHOLDER = re.compile(r"<<([A-Z][A-Z0-9_]*)>>")
LONG_DIGITS = re.compile(r"\d{4,}")

LABEL_GUIDANCE = {
    "PERSON_NAME": "自然人的姓名、别名或收件人姓名",
    "PHONE_NUMBER": "个人手机、座机或分机",
    "EMAIL_ADDRESS": "个人联系邮箱",
    "ADDRESS": "可定位个人的收货、家庭、居住或通信详细地址",
    "DATE_OF_BIRTH": "只允许出生日期或生日，绝不能表示送达、登录、面试或预约日期",
    "CN_RESIDENT_ID": "中国大陆居民身份证号码",
    "PASSPORT_NUMBER": "护照号码",
    "DRIVER_LICENSE_NUMBER": "驾驶证号码",
    "SOCIAL_SECURITY_NUMBER": "个人社保或医保编号",
    "BANK_CARD_NUMBER": "个人银行卡卡号",
    "BANK_ACCOUNT_NUMBER": "个人银行账户号",
    "VEHICLE_LICENSE_PLATE": "个人车辆车牌",
    "EMPLOYEE_ID": "员工工号",
    "STUDENT_ID": "学生学号",
    "MEDICAL_RECORD_NUMBER": "病历号、就诊号或住院号",
    "WECHAT_ID": "微信个人账号",
    "QQ_NUMBER": "QQ个人账号",
    "ALIPAY_ACCOUNT": "支付宝个人账号",
    "USERNAME": "个人登录名或用户名",
    "IP_ADDRESS": "IPv4 或 IPv6 地址",
    "MAC_ADDRESS": "MAC 地址",
    "DEVICE_ID": "IMEI、设备序列号等设备标识",
    "GEO_COORDINATE": "精确经纬度",
    "SECRET": "密码、API key、token 或私钥片段",
}


class CandidateError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt(request: dict[str, Any], count: int) -> str:
    labels = ", ".join(request["labels"])
    guidance = "\n".join(f"- {label}: {LABEL_GUIDANCE[label]}" for label in request["labels"])
    return (
        f"生成恰好 {count} 个彼此显著不同的模板。\n"
        f"批次 ID: {request['batch']}\n"
        f"允许标签: {labels}\n"
        "只允许上述标签，不能引入订单号、快递号、IP 等未列出的占位符。\n"
        f"标签边界定义:\n{guidance}\n"
        f"覆盖领域: {request['domains']}\n"
        "让每个允许标签在本批次尽量均衡出现；至少三分之一模板包含两个以上实体。"
    )


def _extract_json(text: str) -> list[dict[str, Any]]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise CandidateError("model output did not contain a JSON array")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise CandidateError(f"model output JSON error: {exc.msg}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise CandidateError("model output must be an array of objects")
    return value


def _validate_item(
    item: dict[str, Any],
    *,
    batch: str,
    allowed_labels: set[str],
) -> dict[str, Any]:
    if set(item) != {"id", "domain", "scene", "text", "labels"}:
        raise CandidateError(f"{batch}: candidate fields do not match the schema")
    for key in ("id", "domain", "scene", "text"):
        if not isinstance(item[key], str) or not item[key].strip():
            raise CandidateError(f"{batch}: {key} must be a non-empty string")
    if not re.fullmatch(r"[a-z][a-z0-9_]{5,80}", item["id"]):
        raise CandidateError(f"{batch}: invalid candidate id")
    if not isinstance(item["labels"], list) or not item["labels"]:
        raise CandidateError(f"{batch}: labels must be a non-empty array")
    if not all(isinstance(label, str) for label in item["labels"]):
        raise CandidateError(f"{batch}: labels must contain strings")
    declared = set(item["labels"])
    if not declared <= allowed_labels or not declared <= set(CORE_LABELS):
        raise CandidateError(f"{batch}: candidate declared a forbidden label")
    raw_placeholders = RAW_PLACEHOLDER.findall(item["text"])
    if "<<" in RAW_PLACEHOLDER.sub("", item["text"]):
        raise CandidateError(f"{batch}: malformed placeholder")
    if not 1 <= len(raw_placeholders) <= 4:
        raise CandidateError(f"{batch}: template must contain one to four placeholders")
    counts: dict[str, int] = {}

    def normalize(match: re.Match[str]) -> str:
        value = match.group(1)
        numbered = re.fullmatch(r"(.+)_([1-9][0-9]*)", value)
        label = numbered.group(1) if numbered and numbered.group(1) in CORE_LABELS else value
        if label not in CORE_LABELS:
            raise CandidateError(f"{batch}: unknown placeholder label")
        counts[label] = counts.get(label, 0) + 1
        return f"<<{label}_{counts[label]}>>"

    normalized_text = RAW_PLACEHOLDER.sub(normalize, item["text"])
    observed = set(counts)
    if observed != declared:
        raise CandidateError(f"{batch}: labels do not equal placeholder labels")
    if not observed <= allowed_labels:
        raise CandidateError(f"{batch}: placeholder uses a label outside this batch")
    if "DATE_OF_BIRTH" in observed:
        date_context = normalized_text
        if not any(word in date_context for word in ("出生", "生日", "生于")):
            raise CandidateError(f"{batch}: DATE_OF_BIRTH lacks birth context")
    if LONG_DIGITS.search(item["text"]):
        raise CandidateError(f"{batch}: template contains a long literal number")
    if any(word in item["text"] for word in ("测试", "示例", "虚构", "演练", "占位符")):
        raise CandidateError(f"{batch}: template contains a synthetic shortcut word")
    normalized = dict(item)
    normalized["text"] = normalized_text
    normalized["labels"] = sorted(declared)
    normalized["batch"] = batch
    normalized["risk_tiers"] = {label: RISK.get(label, "T1") for label in declared}
    return normalized


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-batch", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument(
        "--only-batch",
        action="append",
        choices=[request["batch"] for request in REQUESTS],
        help="Generate only selected batches; repeat the flag to select several.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 4 <= args.per_batch <= 32:
        raise ValueError("--per-batch must be between 4 and 32")
    model_path = args.model.expanduser().resolve(strict=True)
    if not (model_path / "LICENSE").is_file():
        raise FileNotFoundError("local model is missing LICENSE")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one approved CUDA device before running this script")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    ).eval()
    all_candidates: list[dict[str, Any]] = []
    raw_output_hashes: dict[str, str] = {}
    prompt_hashes: dict[str, str] = {}
    rejection_counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    selected_requests = [
        request
        for request in REQUESTS
        if not args.only_batch or request["batch"] in set(args.only_batch)
    ]
    for request in selected_requests:
        user_prompt = _prompt(request, args.per_batch)
        prompt_hashes[request["batch"]] = hashlib.sha256(
            (SYSTEM_PROMPT + "\n" + user_prompt).encode("utf-8")
        ).hexdigest()
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(generated[0, inputs.shape[1] :], skip_special_tokens=True)
        raw_output_hashes[request["batch"]] = hashlib.sha256(output.encode("utf-8")).hexdigest()
        try:
            items = _extract_json(output)
        except CandidateError:
            _atomic_text(
                args.output.parent / f"{args.output.stem}.{request['batch']}.rejected.txt",
                output,
            )
            raise
        accepted_in_batch = 0
        rejected_in_batch = 0
        for item in items:
            try:
                candidate = _validate_item(
                    item,
                    batch=request["batch"],
                    allowed_labels=set(request["labels"]),
                )
            except CandidateError:
                rejected_in_batch += 1
                continue
            if candidate["id"] in seen_ids or candidate["text"] in seen_texts:
                rejected_in_batch += 1
                continue
            seen_ids.add(candidate["id"])
            seen_texts.add(candidate["text"])
            all_candidates.append(candidate)
            accepted_in_batch += 1
        rejection_counts[request["batch"]] = rejected_in_batch
        if not accepted_in_batch:
            _atomic_text(
                args.output.parent / f"{args.output.stem}.{request['batch']}.rejected.txt",
                output,
            )
            raise CandidateError(f"{request['batch']}: no candidate passed validation")
        print(
            f"generated batch={request['batch']} accepted={accepted_in_batch} "
            f"rejected={rejected_in_batch}"
        )

    model_files = {
        name: _sha256(model_path / name)
        for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")
    }
    payload = {
        "schema_version": 1,
        "status": "quarantined_candidates",
        "created_at": datetime.now(UTC).isoformat(),
        "generator": {
            "model_id": "Qwen/Qwen3-8B",
            "model_path_redacted": model_path.name,
            "declared_license": "Apache-2.0",
            "file_hashes": model_files,
            "prompt_hashes": prompt_hashes,
            "raw_output_hashes": raw_output_hashes,
            "rejection_counts": rejection_counts,
            "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        },
        "candidates": all_candidates,
    }
    _atomic_json(args.output, payload)
    print(f"candidate_count={len(all_candidates)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
