"""Deterministic, visibly synthetic value generators.

Structured identifiers use reserved test sentinels (``990000`` for resident
IDs and ``999999`` for card IINs).  They can exercise checksum validators but
must never be interpreted as issued credentials.
"""

from __future__ import annotations

import random
import string
from datetime import date, timedelta

from ..validators import cn_resident_id_check_code, luhn_check_digit

SYNTHETIC_ID_REGION_PREFIX = "990000"
SYNTHETIC_CARD_IIN = "999999"
_TOKEN_ALPHABET = "0123456789abcdefghjkmnpqruvwxyz"
LEGACY_VALUE_VARIANT = "legacy"
_SINGLE_SURNAMES = tuple(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳"
)
_COMPOUND_SURNAMES = (
    "欧阳",
    "司马",
    "上官",
    "诸葛",
    "夏侯",
    "东方",
    "皇甫",
    "尉迟",
    "公孙",
    "慕容",
    "司徒",
    "端木",
    "令狐",
    "钟离",
    "宇文",
    "长孙",
    "南宫",
    "独孤",
    "百里",
    "申屠",
)
_GIVEN_NAME_CHARACTERS = tuple(
    "安澜清和宁川云汀星野知远景明怀瑾若谷青禾映昭嘉木砚舟时雨长风予初南乔书言修竹沐辰令仪望舒庭宇千帆月白松遥亦航茂树闻溪照原疏桐"
)
_ADDRESS_PROVINCES = ("宁川省", "澄海省", "云岭省", "栖原省", "沧澜省", "星泽省")
_ADDRESS_CITIES = ("云汀市", "清和市", "望舒市", "星野市", "松遥市", "闻溪市", "景明市", "月白市")
_ADDRESS_DISTRICTS = (
    "栖霞区",
    "砚舟区",
    "青禾区",
    "嘉木区",
    "长风区",
    "昭宁区",
    "修竹区",
    "千帆区",
)
_ADDRESS_TOWNS = ("南乔镇", "疏桐镇", "沐辰乡", "令仪乡", "庭宇镇", "照野乡")
_ADDRESS_ROADS = (
    "清和路",
    "砚池路",
    "嘉树街",
    "望舒大道",
    "松风路",
    "闻溪街",
    "星河路",
    "青禾大道",
)


class SyntheticValueFactory:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self._serials: dict[str, int] = {}
        self._offsets: dict[str, int] = {}

    def _unique_number(self, key: str, modulus: int) -> int:
        """Return a deterministic non-repeating value until ``modulus`` is exhausted."""

        if modulus < 1:
            raise ValueError("modulus must be positive")
        if key not in self._offsets:
            self._offsets[key] = self.rng.randrange(modulus)
        offset = self._offsets[key]
        serial = self._serials.get(key, 0)
        self._serials[key] = serial + 1
        return (offset + serial) % modulus

    def _digits(self, length: int) -> str:
        return "".join(self.rng.choice(string.digits) for _ in range(length))

    def _unique_digits(self, key: str, length: int) -> str:
        return f"{self._unique_number(key, 10**length):0{length}d}"

    def _token(self, length: int = 8) -> str:
        return "".join(self.rng.choice(_TOKEN_ALPHABET) for _ in range(length))

    def _unique_token(self, key: str, length: int) -> str:
        value = self._unique_number(key, len(_TOKEN_ALPHABET) ** length)
        encoded = []
        for _ in range(length):
            value, remainder = divmod(value, len(_TOKEN_ALPHABET))
            encoded.append(_TOKEN_ALPHABET[remainder])
        return "".join(reversed(encoded))

    def _unique_product(self, key: str, alphabets: tuple[tuple[str, ...], ...]) -> str:
        modulus = 1
        for alphabet in alphabets:
            if not alphabet:
                raise ValueError("product alphabets cannot be empty")
            modulus *= len(alphabet)
        value = self._unique_number(key, modulus)
        selected: list[str] = []
        for alphabet in reversed(alphabets):
            value, remainder = divmod(value, len(alphabet))
            selected.append(alphabet[remainder])
        return "".join(reversed(selected))

    def person_name(self) -> str:
        # No list of real full names is read.  Ten disjoint style streams
        # construct unique two-to-four-character names from independent
        # surname/given-name components, with one small mixed-script stream.
        style = self._unique_number("person_name_style", 10)
        if style < 2:
            surnames = _SINGLE_SURNAMES[style::2]
            return self._unique_product(
                f"person_name_{style}",
                (surnames, _GIVEN_NAME_CHARACTERS),
            )
        if style < 5:
            surnames = _SINGLE_SURNAMES[(style - 2) :: 3]
            return self._unique_product(
                f"person_name_{style}",
                (surnames, _GIVEN_NAME_CHARACTERS, _GIVEN_NAME_CHARACTERS),
            )
        if style == 5:
            return self._unique_product(
                "person_name_5",
                (_COMPOUND_SURNAMES, _GIVEN_NAME_CHARACTERS),
            )
        if style < 8:
            surnames = _COMPOUND_SURNAMES[(style - 6) :: 2]
            return self._unique_product(
                f"person_name_{style}",
                (surnames, _GIVEN_NAME_CHARACTERS, _GIVEN_NAME_CHARACTERS),
            )
        if style == 8:
            return self._unique_product(
                "person_name_8",
                (
                    _SINGLE_SURNAMES,
                    _GIVEN_NAME_CHARACTERS,
                    _GIVEN_NAME_CHARACTERS,
                    _GIVEN_NAME_CHARACTERS,
                ),
            )
        surname = self._unique_product("person_name_9_surname", (_SINGLE_SURNAMES,))
        return surname + self._unique_token("person_name_9_latin", 6).capitalize()

    def phone(self) -> str:
        # Clearly identified as synthetic by every enclosing template/provenance.
        return f"199{self._unique_number('phone', 100_000_000):08d}"

    def email(self) -> str:
        # RFC 2606 reserves .invalid for examples.
        serial = self._unique_number("email_style", 3)
        domains = ("example.invalid", "mail.invalid", "contact.invalid")
        separators = ("-", ".", "_")
        return f"pii{separators[serial]}{self._unique_token('email', 10)}@{domains[serial]}"

    def address(self) -> str:
        serial = self._unique_number("address", 10_000_000)
        number = serial + 1
        province = _ADDRESS_PROVINCES[serial % len(_ADDRESS_PROVINCES)]
        city = _ADDRESS_CITIES[(serial // 3) % len(_ADDRESS_CITIES)]
        district = _ADDRESS_DISTRICTS[(serial // 5) % len(_ADDRESS_DISTRICTS)]
        town = _ADDRESS_TOWNS[(serial // 7) % len(_ADDRESS_TOWNS)]
        road = _ADDRESS_ROADS[(serial // 11) % len(_ADDRESS_ROADS)]
        style = serial % 8
        if style == 0:
            return f"{province}{city}{district}{road}{number}号"
        if style == 1:
            return f"{city}{district}{road}{number}号{serial % 31 + 1}栋"
        if style == 2:
            return f"{province}{city}{town}清溪村{number}号"
        if style == 3:
            return f"{city}{district}{town}{road}{serial % 99 + 1}弄{number}室"
        if style == 4:
            return f"{province}{city}{district}嘉禾园{number}号楼"
        if style == 5:
            return f"{city}{town}{road}{number}号{serial % 12 + 1}单元"
        if style == 6:
            return f"{province}{district}{road}与闻溪街交汇处{number}座"
        return f"{city}{district}千帆产业园{road}{number}号"

    def date_of_birth(self) -> str:
        origin = date(1980, 1, 1)
        offset = self._unique_number("date_of_birth", 30 * 365)
        value = origin + timedelta(days=offset)
        style = offset % 3
        if style == 0:
            return value.isoformat()
        if style == 1:
            return f"{value.year}年{value.month}月{value.day}日"
        return f"{value.year}/{value.month:02d}/{value.day:02d}"

    def cn_resident_id(self) -> str:
        origin = date(1980, 1, 1)
        birth = origin + timedelta(days=self._unique_number("resident_birth", 30 * 365))
        sequence = f"{self._unique_number('resident_sequence', 999) + 1:03d}"
        first_seventeen = SYNTHETIC_ID_REGION_PREFIX + birth.strftime("%Y%m%d") + sequence
        return first_seventeen + cn_resident_id_check_code(first_seventeen)

    def passport(self) -> str:
        return "PZ" + self._unique_token("passport", 8).upper()

    def driver_license(self) -> str:
        return "DL" + self._unique_token("driver_license", 10).upper()

    def social_security(self) -> str:
        return "990000" + self._unique_digits("social_security", 12)

    def bank_card(self) -> str:
        suffix = self._unique_number("bank_card", 1_000_000_000)
        without_check = SYNTHETIC_CARD_IIN + f"{suffix:09d}"
        return without_check + luhn_check_digit(without_check)

    def bank_account(self) -> str:
        return "9900" + self._unique_digits("bank_account", 14)

    def vehicle_plate(self) -> str:
        return "岚A·" + self._unique_token("vehicle_plate", 7).upper()

    def employee_id(self) -> str:
        return "E" + self._unique_digits("employee_id", 10)

    def student_id(self) -> str:
        return "S" + self._unique_digits("student_id", 12)

    def medical_record(self) -> str:
        return "M" + self._unique_digits("medical_record", 12)

    def wechat_id(self) -> str:
        return "wx_" + self._unique_token("wechat_id", 12)

    def qq_number(self) -> str:
        return "9" + self._unique_digits("qq_number", 11)

    def alipay_account(self) -> str:
        return f"pay-{self._unique_token('alipay_account', 10)}@example.invalid"

    def username(self) -> str:
        return "u_" + self._unique_token("username", 12)

    def ip_address(self) -> str:
        # 2001:db8::/32 is reserved for documentation by RFC 3849.  The
        # 64-bit serial avoids joining otherwise independent template groups
        # when a large corpus is split by normalized entity value.
        suffix = f"{self._unique_number('ip_address', 2**64):016x}"
        hextets = ":".join(suffix[index : index + 4] for index in range(0, 16, 4))
        return f"2001:db8::{hextets}"

    def mac_address(self) -> str:
        # Locally administered unicast prefix.
        suffix = self._unique_number("mac_address", 2**24)
        return "02:00:00:" + ":".join(f"{(suffix >> shift) & 0xFF:02X}" for shift in (16, 8, 0))

    def device_id(self) -> str:
        return "DEV-" + self._unique_token("device_id", 14).upper()

    def coordinate(self) -> str:
        # Deterministic points in a small synthetic grid; every serial maps to
        # a distinct six-decimal pair for the first 4e12 values.
        serial = self._unique_number("coordinate", 4_000_000_000_000)
        latitude = (serial % 2_000_000) / 1_000_000 - 1.0
        longitude = (serial // 2_000_000) / 1_000_000 - 1.0
        return f"{latitude:.6f},{longitude:.6f}"

    def secret(self) -> str:
        return "key_" + self._unique_token("secret", 24)

    def order_id(self) -> str:
        return "ORD-" + self._unique_digits("order_id", 12)

    def trace_id(self) -> str:
        return "TRACE-" + self._unique_token("trace_id", 18)

    def model_version(self) -> str:
        return f"model-v{self.rng.randint(1, 9)}.{self.rng.randint(0, 9)}.{self.rng.randint(0, 9)}"

    def invalid_resident_id(self) -> str:
        valid = self.cn_resident_id()
        replacement = "0" if valid[-1] != "0" else "1"
        return valid[:-1] + replacement

    def masked_phone(self) -> str:
        return "199****" + self._digits(4)

    def transaction_id(self) -> str:
        return "TXN-" + self._unique_token("transaction_id", 16).upper()

    def invoice_code(self) -> str:
        return "INV-" + self._unique_digits("invoice_code", 12)

    def product_code(self) -> str:
        return "SKU-" + self._unique_token("product_code", 12).upper()

    def organization(self) -> str:
        return "云汀数理研究社-" + self._unique_token("organization", 10)

    def job_title(self) -> str:
        return "数据协调员-" + self._unique_token("job_title", 8).upper()

    def generic_date(self) -> str:
        origin = date(2090, 1, 1)
        value = origin + timedelta(days=self._unique_number("generic_date", 10 * 365))
        return value.isoformat()

    def organization_address(self) -> str:
        number = self._unique_number("organization_address", 1_000_000) + 1
        return f"宁川省云汀市清和园区砚池路{number}号"

    def quad_version(self) -> str:
        return "v" + ".".join(str(self.rng.randint(0, 9)) for _ in range(4))

    def invalid_card(self) -> str:
        valid = self.bank_card()
        replacement = "0" if valid[-1] != "0" else "1"
        return valid[:-1] + replacement

    def region_code(self) -> str:
        return "RGN-" + self._unique_digits("region_code", 8)

    def vehicle_plate_fictional_compact(self) -> str:
        return "岚B" + self._unique_token("vehicle_plate_fictional_compact", 5).upper()

    def vehicle_plate_fictional_hyphenated(self) -> str:
        return "岚C-" + self._unique_token("vehicle_plate_fictional_hyphenated", 6).upper()

    def vehicle_plate_fictional_extended(self) -> str:
        return "岚D·" + self._unique_token("vehicle_plate_fictional_extended", 8).upper()

    def employee_id_department_hyphenated(self) -> str:
        return "DPT-ZZ-" + self._unique_digits("employee_id_department_hyphenated", 8)

    def employee_id_staff_slash(self) -> str:
        return "STAFF/ZZ/" + self._unique_digits("employee_id_staff_slash", 8)

    def date_of_birth_compact_yyyymmdd(self) -> str:
        origin = date(1980, 1, 1)
        value = origin + timedelta(
            days=self._unique_number("date_of_birth_compact_yyyymmdd", 30 * 365)
        )
        return value.strftime("%Y%m%d")

    def date_of_birth_dot_separated(self) -> str:
        origin = date(1980, 1, 1)
        value = origin + timedelta(
            days=self._unique_number("date_of_birth_dot_separated", 30 * 365)
        )
        return value.strftime("%Y.%m.%d")

    def driver_license_fictional_hyphenated(self) -> str:
        return "DL-ZZ-" + self._unique_token("driver_license_fictional_hyphenated", 8).upper()

    def coordinate_signed_spaced(self) -> str:
        serial = self._unique_number("coordinate_signed_spaced", 4_000_000_000_000)
        latitude = (serial % 2_000_000) / 1_000_000 - 1.0
        longitude = (serial // 2_000_000) / 1_000_000 - 1.0
        return f"{latitude:+.6f}, {longitude:+.6f}"

    def mac_address_hyphen_lowercase(self) -> str:
        suffix = self._unique_number("mac_address_hyphen_lowercase", 2**24)
        return "02-00-00-" + "-".join(f"{(suffix >> shift) & 0xFF:02x}" for shift in (16, 8, 0))

    def passport_fictional_hyphenated(self) -> str:
        return "PZ-" + self._unique_token("passport_fictional_hyphenated", 8).upper()

    def build_id(self) -> str:
        return "BLD-" + self._unique_digits("build_id", 12)

    def config_key_name(self) -> str:
        return "auth.token.rotation_policy." + self._unique_digits("config_key_name", 6)

    def release_date(self) -> str:
        origin = date(2030, 1, 1)
        value = origin + timedelta(days=self._unique_number("release_date", 20 * 365))
        return value.isoformat()

    def device_model(self) -> str:
        return "EDGE-MODEL-" + self._unique_token("device_model", 8).upper()


GENERATOR_METHODS = {
    "PERSON_NAME": "person_name",
    "PHONE_NUMBER": "phone",
    "EMAIL_ADDRESS": "email",
    "ADDRESS": "address",
    "DATE_OF_BIRTH": "date_of_birth",
    "CN_RESIDENT_ID": "cn_resident_id",
    "PASSPORT_NUMBER": "passport",
    "DRIVER_LICENSE_NUMBER": "driver_license",
    "SOCIAL_SECURITY_NUMBER": "social_security",
    "BANK_CARD_NUMBER": "bank_card",
    "BANK_ACCOUNT_NUMBER": "bank_account",
    "VEHICLE_LICENSE_PLATE": "vehicle_plate",
    "EMPLOYEE_ID": "employee_id",
    "STUDENT_ID": "student_id",
    "MEDICAL_RECORD_NUMBER": "medical_record",
    "WECHAT_ID": "wechat_id",
    "QQ_NUMBER": "qq_number",
    "ALIPAY_ACCOUNT": "alipay_account",
    "USERNAME": "username",
    "IP_ADDRESS": "ip_address",
    "MAC_ADDRESS": "mac_address",
    "DEVICE_ID": "device_id",
    "GEO_COORDINATE": "coordinate",
    "SECRET": "secret",
    "ORDER_ID": "order_id",
    "TRACE_ID": "trace_id",
    "MODEL_VERSION": "model_version",
    "INVALID_CN_ID": "invalid_resident_id",
    "MASKED_PHONE": "masked_phone",
    "TRANSACTION_ID": "transaction_id",
    "INVOICE_CODE": "invoice_code",
    "PRODUCT_CODE": "product_code",
    "ORGANIZATION": "organization",
    "JOB_TITLE": "job_title",
    "GENERIC_DATE": "generic_date",
    "ORG_ADDRESS": "organization_address",
    "QUAD_VERSION": "quad_version",
    "INVALID_CARD": "invalid_card",
    "REGION_CODE": "region_code",
    "BUILD_ID": "build_id",
    "CONFIG_KEY_NAME": "config_key_name",
    "RELEASE_DATE": "release_date",
    "DEVICE_MODEL": "device_model",
}

VALUE_VARIANT_METHODS = {
    ("VEHICLE_LICENSE_PLATE", "fictional_compact"): "vehicle_plate_fictional_compact",
    ("VEHICLE_LICENSE_PLATE", "fictional_hyphenated"): "vehicle_plate_fictional_hyphenated",
    ("VEHICLE_LICENSE_PLATE", "fictional_extended"): "vehicle_plate_fictional_extended",
    ("EMPLOYEE_ID", "department_hyphenated"): "employee_id_department_hyphenated",
    ("EMPLOYEE_ID", "staff_slash"): "employee_id_staff_slash",
    ("DATE_OF_BIRTH", "compact_yyyymmdd"): "date_of_birth_compact_yyyymmdd",
    ("DATE_OF_BIRTH", "dot_separated"): "date_of_birth_dot_separated",
    (
        "DRIVER_LICENSE_NUMBER",
        "fictional_hyphenated",
    ): "driver_license_fictional_hyphenated",
    ("GEO_COORDINATE", "signed_spaced"): "coordinate_signed_spaced",
    ("MAC_ADDRESS", "hyphen_lowercase"): "mac_address_hyphen_lowercase",
    ("PASSPORT_NUMBER", "fictional_hyphenated"): "passport_fictional_hyphenated",
}


def supports_value_variant(generator_key: str, value_variant: str) -> bool:
    return (
        value_variant == LEGACY_VALUE_VARIANT
        or (
            generator_key,
            value_variant,
        )
        in VALUE_VARIANT_METHODS
    )


def generate_value(
    factory: SyntheticValueFactory,
    generator_key: str,
    *,
    value_variant: str = LEGACY_VALUE_VARIANT,
) -> str:
    if value_variant != LEGACY_VALUE_VARIANT:
        try:
            method_name = VALUE_VARIANT_METHODS[(generator_key, value_variant)]
        except KeyError as exc:
            raise KeyError(
                f"unknown synthetic value variant: {generator_key}:{value_variant}"
            ) from exc
        return str(getattr(factory, method_name)())
    try:
        method_name = GENERATOR_METHODS[generator_key]
    except KeyError as exc:
        raise KeyError(f"unknown synthetic generator key: {generator_key}") from exc
    return str(getattr(factory, method_name)())
