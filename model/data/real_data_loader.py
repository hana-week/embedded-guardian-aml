"""
금융결제원 D-테스트베드 실데이터 로더
현장에서 이 파일만 수정하면 Layer 1~2 그대로 동작

지원 테이블:
  HF_TRNS_TRAN  — 홈·펌뱅킹 이체 (핵심)
  OB_TRNS_TRAN  — 오픈뱅킹 이체 (거래소 링킹 핵심)
  CD_TRNS_TRAN  — CD기(ATM) 거래
  PI_WD_LEDG    — 개인 출금 원장 (인구통계)
  GR_JC_TRAN    — 지로 자동이체
  CMS_REQ_TRAN  — CMS 출금 요청

결합 Key: (금융회사일련번호 FC_SN + 계좌일련번호 AC_SN)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# ── 현장에서 실제 파일 경로로 수정 ──────────────────────
DATA_ROOT = Path(__file__).parent / "kftc"   # 금융결제원 데이터 폴더

# 파일명 (현장 환경에 맞게 수정 필요)
TABLE_FILES = {
    "HF_TRNS_TRAN": DATA_ROOT / "HF_TRNS_TRAN.csv",
    "OB_TRNS_TRAN": DATA_ROOT / "OB_TRNS_TRAN.csv",
    "CD_TRNS_TRAN": DATA_ROOT / "CD_TRNS_TRAN.csv",
    "PI_WD_LEDG":   DATA_ROOT / "PI_WD_LEDG.csv",
    "GR_JC_TRAN":   DATA_ROOT / "GR_JC_TRAN.csv",
    "CMS_REQ_TRAN": DATA_ROOT / "CMS_REQ_TRAN.csv",
}

# ── 컬럼 매핑 (현장에서 실제 컬럼명 확인 후 수정) ────────
# 형식: "실데이터 컬럼명": "내부 사용 컬럼명"
# !! 현장에서 반드시 명세서와 대조 확인 필요 !!

HF_COL_MAP = {
    # 송금 계좌 식별자
    "WD_FI_CD":    "sender_fi",      # 출금 금융회사 코드
    "WD_AC_SN":    "sender_id",      # 출금 계좌 일련번호
    # 수신 계좌 식별자
    "DPS_FI_CD":   "receiver_fi",    # 입금 금융회사 코드
    "DPS_AC_SN":   "receiver_id",    # 입금 계좌 일련번호
    # 거래 정보
    "TRAN_AMT":    "amount",         # 거래 금액
    "TRAN_DT":     "date",           # 거래 일자 (YYYYMMDD)
    "TRAN_TM":     "tran_tm",        # 거래 시각 (HHMMSS) → hour 파생
    # AML 라벨
    "FF_SP_AI":    "is_suspicious",  # 의심거래 플래그 (핵심 라벨!)
}

OB_COL_MAP = {
    "WD_FI_CD":    "sender_fi",
    "WD_AC_SN":    "sender_id",
    "DPS_FI_CD":   "receiver_fi",
    "DPS_AC_SN":   "receiver_id",
    "TRAN_AMT":    "amount",
    "TRAN_DT":     "date",
    "TRAN_TMRG":   "tran_tm",        # 오픈뱅킹은 TRAN_TMRG
    "UO_SN":       "uo_sn",          # 이용기관 일련번호 (거래소 식별 핵심!)
    "UO_NAME":     "uo_name",        # 이용기관명
}

CD_COL_MAP = {
    "FI_CD":       "sender_fi",
    "AC_SN":       "sender_id",
    "TRAN_AMT":    "amount",
    "TRAN_DT":     "date",
    "TRAN_TM":     "tran_tm",
}

PI_COL_MAP = {
    "FI_CD":       "fi_cd",
    "AC_SN":       "account_id",
    "AGE_RNGE":    "age_group",      # 연령대
    "SEX_CD":      "gender",         # 성별 코드
}

# 가상자산 거래소 이용기관 키워드 (크로스체인 링킹용)
# !! 현장에서 실제 OB_TRNS_TRAN의 UO_NAME 확인 후 보완 !!
EXCHANGE_KEYWORDS = [
    "업비트", "빗썸", "코인원", "코빗", "고팍스",
    "UPBIT", "BITHUMB", "COINONE", "KORBIT", "GOPAX",
]


# ── 테이블 로딩 ──────────────────────────────────────────

def _load_table(table_name: str, col_map: dict,
                encoding: str = "cp949") -> Optional[pd.DataFrame]:
    path = TABLE_FILES[table_name]
    if not path.exists():
        print(f"  [SKIP] {table_name} 파일 없음: {path}")
        return None

    df = pd.read_csv(path, encoding=encoding, low_memory=False)
    print(f"  [OK]   {table_name}: {len(df):,}행 로딩")

    # 실제 존재하는 컬럼만 매핑
    existing = {k: v for k, v in col_map.items() if k in df.columns}
    missing = [k for k in col_map if k not in df.columns]
    if missing:
        print(f"         !! 매핑 못한 컬럼 (현장 확인 필요): {missing}")

    df = df.rename(columns=existing)
    return df


def _parse_hour(tran_tm_series: pd.Series) -> pd.Series:
    """HHMMSS 형식 시각 → 시(0-23) 추출"""
    s = tran_tm_series.astype(str).str.zfill(6)
    return s.str[:2].astype(int, errors="ignore").fillna(0).astype(int)


def _make_account_id(fi_col: pd.Series, ac_col: pd.Series) -> pd.Series:
    """금융회사코드 + 계좌일련번호 결합 → 고유 계좌 ID"""
    return fi_col.astype(str).str.zfill(4) + "_" + ac_col.astype(str)


# ── 메인 로딩 함수 ──────────────────────────────────────

def load_transactions() -> pd.DataFrame:
    """
    모든 이체 테이블을 통합하여 transactions DataFrame 반환
    columns: txn_id, sender_id, receiver_id, amount, date, hour,
             txn_type, aml_label, [uo_sn, uo_name]
    """
    print("\n[이체 데이터 로딩]")
    frames = []

    # 1. 홈·펌뱅킹 이체
    hf = _load_table("HF_TRNS_TRAN", HF_COL_MAP)
    if hf is not None:
        hf["txn_type"] = "transfer"
        hf["aml_label"] = hf.get("is_suspicious", 0).fillna(0).astype(int)
        frames.append(hf)

    # 2. 오픈뱅킹 이체 (거래소 링킹 핵심)
    ob = _load_table("OB_TRNS_TRAN", OB_COL_MAP)
    if ob is not None:
        ob["txn_type"] = "openbank"
        ob["aml_label"] = 0  # OB에는 FF_SP_AI 없음 — GNN이 탐지
        # 거래소 방향 이체 표시 (크로스체인 링킹 Step 1)
        if "uo_name" in ob.columns:
            is_exchange = ob["uo_name"].str.contains(
                "|".join(EXCHANGE_KEYWORDS), na=False, case=False
            )
            ob.loc[is_exchange, "aml_label"] = -1  # -1: 거래소 방향 (별도 분석)
            print(f"         거래소 방향 이체: {is_exchange.sum():,}건 식별")
        frames.append(ob)

    # 3. ATM 거래 (sender만 있음)
    cd = _load_table("CD_TRNS_TRAN", CD_COL_MAP)
    if cd is not None:
        cd["txn_type"] = "atm"
        cd["aml_label"] = 0
        cd["receiver_id"] = "ATM"  # ATM 출금은 수신자 없음
        frames.append(cd)

    if not frames:
        raise FileNotFoundError(
            f"이체 데이터 파일 없음. {DATA_ROOT} 경로 확인 필요"
        )

    txn = pd.concat(frames, ignore_index=True)

    # sender_id / receiver_id 정규화
    for col in ["sender_id", "receiver_id"]:
        fi_col = col.replace("_id", "_fi")
        if fi_col in txn.columns:
            txn[col] = _make_account_id(txn[fi_col], txn[col])

    # 날짜 파싱
    txn["date"] = pd.to_datetime(txn["date"].astype(str), format="%Y%m%d", errors="coerce")

    # 시각 → hour
    if "tran_tm" in txn.columns:
        txn["hour"] = _parse_hour(txn["tran_tm"])
    else:
        txn["hour"] = 12  # 시각 정보 없으면 기본값

    # txn_id 생성
    txn = txn.reset_index(drop=True)
    txn["txn_id"] = ["TXN" + str(i).zfill(8) for i in txn.index]

    # 필수 컬럼만 선택
    keep = ["txn_id", "sender_id", "receiver_id", "amount",
            "date", "hour", "txn_type", "aml_label"]
    optional = ["uo_sn", "uo_name"]
    keep += [c for c in optional if c in txn.columns]

    txn = txn[[c for c in keep if c in txn.columns]]
    txn["amount"] = pd.to_numeric(txn["amount"], errors="coerce").fillna(0)

    print(f"\n  통합 거래: {len(txn):,}건")
    aml = (txn["aml_label"] == 1).sum()
    exch = (txn["aml_label"] == -1).sum()
    print(f"  FF_SP_AI 플래그: {aml:,}건 | 거래소 방향: {exch:,}건")
    return txn


def load_accounts(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    PI_WD_LEDG 인구통계 + 거래 데이터에서 계좌 목록 생성
    columns: account_id, gender, age_group, [credit_grade, platform_risk,
             num_accounts, dawn_txn_ratio, easy_pay_cnt], is_suspicious
    """
    print("\n[계좌 데이터 구성]")

    # 거래에 등장하는 모든 계좌 ID 수집
    all_ids = pd.Series(
        pd.concat([transactions["sender_id"], transactions["receiver_id"]])
        .unique()
    ).rename("account_id")
    accounts = pd.DataFrame({"account_id": all_ids})

    # PI_WD_LEDG 인구통계 합류
    pi = _load_table("PI_WD_LEDG", PI_COL_MAP)
    if pi is not None:
        pi["account_id"] = _make_account_id(pi["fi_cd"], pi["account_id"])
        accounts = accounts.merge(
            pi[["account_id", "age_group", "gender"]],
            on="account_id", how="left"
        )
    else:
        accounts["age_group"] = 40
        accounts["gender"] = 1

    # 기본값 (NICE 데이터 없을 경우)
    accounts["credit_grade"] = 5
    accounts["platform_risk"] = 2
    accounts["num_accounts"] = 1
    accounts["dawn_txn_ratio"] = 0.0
    accounts["easy_pay_cnt"] = 5.0
    accounts["balance"] = 0.0

    # AML 라벨: FF_SP_AI 기반
    flagged_accounts = set(
        transactions[transactions["aml_label"] == 1]["sender_id"].tolist() +
        transactions[transactions["aml_label"] == 1]["receiver_id"].tolist()
    )
    accounts["is_suspicious"] = accounts["account_id"].isin(flagged_accounts).astype(int)

    accounts = accounts.fillna({
        "age_group": 40, "gender": 1,
        "credit_grade": 5, "platform_risk": 2,
    })

    susp = accounts["is_suspicious"].sum()
    print(f"  계좌: {len(accounts):,}개 | FF_SP_AI 관련: {susp:,}개")
    return accounts.reset_index(drop=True)


def load_real_data():
    """
    실데이터 로드 진입점.
    Returns: (accounts_df, transactions_df)
    synthetic_data.py의 generate_accounts/generate_transactions 대체
    """
    print("=" * 50)
    print("금융결제원 D-테스트베드 실데이터 로딩")
    print(f"데이터 경로: {DATA_ROOT}")
    print("=" * 50)

    transactions = load_transactions()
    accounts = load_accounts(transactions)

    print("\n로딩 완료.")
    return accounts, transactions


# ── 현장 EDA 빠른 확인 ──────────────────────────────────

def quick_check():
    """현장 도착 직후 실행: 데이터 구조 5분 확인"""
    print("\n[빠른 데이터 검증]")
    accounts, transactions = load_real_data()

    print("\n-- accounts --")
    print(accounts.dtypes)
    print(accounts.head(3).to_string())

    print("\n-- transactions --")
    print(transactions.dtypes)
    print(transactions.head(3).to_string())

    print("\n-- 날짜 범위 --")
    print(transactions["date"].min(), "~", transactions["date"].max())

    print("\n-- 금액 분포 --")
    print(transactions["amount"].describe())

    missing = transactions.isnull().sum()
    if missing.any():
        print("\n-- 결측치 --")
        print(missing[missing > 0])


if __name__ == "__main__":
    # 현장에서 실행: python real_data_loader.py
    quick_check()
