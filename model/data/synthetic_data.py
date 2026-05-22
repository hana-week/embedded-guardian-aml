"""
합성 이체 트랜잭션 데이터 생성기
- NICE평가정보 분포 기반 계좌 노드 생성
- BC카드/NH농협 통계 기반 거래 특성 반영
- AML 시나리오(smurfing, layering) 주입
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_ACCOUNTS = 5_000
N_TRANSACTIONS = 30_000
OUTPUT_DIR = Path(__file__).parent


def generate_accounts(n=N_ACCOUNTS) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    age_groups = [19, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90]
    age_weights = [0.03, 0.07, 0.09, 0.12, 0.13, 0.12, 0.11, 0.10, 0.08, 0.06, 0.04, 0.03, 0.01, 0.01, 0.00]

    df = pd.DataFrame({
        "account_id": [f"ACC{i:05d}" for i in range(n)],
        "gender": rng.choice([1, 2], size=n, p=[0.52, 0.48]),
        "age_group": rng.choice(age_groups, size=n, p=age_weights),
        "region": rng.choice(range(1, 12), size=n),
        # NICE 신용위험평점: 1등급(최고위험) ~ 10등급(최저위험)
        "credit_grade": rng.choice(range(1, 11), size=n,
                                    p=[0.03, 0.07, 0.10, 0.13, 0.15, 0.15, 0.14, 0.12, 0.08, 0.03]),
        # 플랫폼리스크 ML 스코어: 0(저위험) ~ 10(고위험)
        "platform_risk": rng.choice(range(11), size=n,
                                     p=[0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.01, 0.005, 0.005]),
        # NH농협 자유입출식 계좌 수
        "num_accounts": rng.choice(range(1, 8), size=n,
                                    p=[0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]),
        # BC카드 새벽(00-05시) 소비 비율
        "dawn_txn_ratio": rng.beta(1, 10, size=n).round(4),
        # BC카드 간편결제 월평균 이용 건수
        "easy_pay_cnt": rng.poisson(8, size=n).astype(float),
        # 계좌 잔액 (원)
        "balance": rng.lognormal(mean=6, sigma=1.5, size=n).round(0),
    })

    # Proxy 라벨: FF_SP_AI 대체
    # 플랫폼리스크 최고위험(>=8) OR (신용 최하등급<=2 AND 다계좌>=4)
    df["is_suspicious"] = (
        (df["platform_risk"] >= 8) |
        ((df["credit_grade"] <= 2) & (df["num_accounts"] >= 4))
    ).astype(int)

    return df


def generate_transactions(accounts_df: pd.DataFrame, n=N_TRANSACTIONS) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    account_ids = accounts_df["account_id"].values
    n_acc = len(account_ids)

    sender_idx = rng.integers(0, n_acc, size=n)
    receiver_idx = (sender_idx + rng.integers(1, n_acc, size=n)) % n_acc

    # 실제 오픈뱅킹 시간대 분포 반영 (합계 1이 되도록 정규화)
    hour_weights = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                    0.03, 0.05, 0.07, 0.07, 0.06, 0.06,
                    0.06, 0.07, 0.07, 0.06, 0.06, 0.06,
                    0.06, 0.05, 0.04, 0.04, 0.03, 0.02]
    hw = np.array(hour_weights, dtype=float)
    hw /= hw.sum()
    hours = rng.choice(range(24), size=n, p=hw)

    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    txn_dates = rng.choice(dates, size=n)

    df = pd.DataFrame({
        "txn_id": [f"TXN{i:07d}" for i in range(n)],
        "sender_id": account_ids[sender_idx],
        "receiver_id": account_ids[receiver_idx],
        # lognormal(mean=13, sigma=1.5) → 중앙값 약 44만원, 한국 은행 이체 현실적 분포
        "amount": rng.lognormal(mean=13, sigma=1.5, size=n).round(0),
        "date": txn_dates,
        "hour": hours,
        "txn_type": rng.choice(
            ["transfer", "atm", "openbank", "cms"],
            size=n, p=[0.45, 0.20, 0.30, 0.05]
        ),
        "aml_label": 0,
    })

    df = _inject_smurfing(df, accounts_df, rng)
    df = _inject_layering(df, accounts_df, rng)

    return df.reset_index(drop=True)


def _inject_smurfing(df: pd.DataFrame, accounts_df: pd.DataFrame, rng) -> pd.DataFrame:
    """소액 분산 입금(Smurfing): 2천만원 임계치 아래 반복 분산 입금"""
    suspicious = accounts_df[accounts_df["is_suspicious"] == 1]["account_id"].values
    if len(suspicious) == 0:
        return df

    new_rows = []
    for _ in range(200):
        target = rng.choice(suspicious)
        n_splits = int(rng.integers(3, 8))
        total = rng.uniform(15_000_000, 19_800_000)
        # 같은 날 분산 입금이어야 R3 규칙이 탐지 가능
        txn_date = rng.choice(pd.date_range("2025-01-01", periods=365, freq="D"))

        for _ in range(n_splits):
            new_rows.append({
                "txn_id": f"TXN_SMF_{len(new_rows):06d}",
                "sender_id": f"ACC{rng.integers(0, N_ACCOUNTS):05d}",
                "receiver_id": target,
                "amount": round(total / n_splits * rng.uniform(0.9, 1.1), 0),
                "date": txn_date,
                "hour": int(rng.integers(0, 24)),
                "txn_type": "transfer",
                "aml_label": 1,
            })

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def _inject_layering(df: pd.DataFrame, accounts_df: pd.DataFrame, rng) -> pd.DataFrame:
    """레이어링: A->B->C->D 연속 이체로 추적 회피"""
    suspicious = accounts_df[accounts_df["is_suspicious"] == 1]["account_id"].values
    if len(suspicious) == 0:
        return df

    new_rows = []
    for _ in range(60):
        chain_len = int(rng.integers(3, 6))
        chain = [str(rng.choice(suspicious)) for _ in range(chain_len + 1)]
        amount = rng.uniform(50_000_000, 200_000_000)
        base_date = rng.choice(pd.date_range("2025-01-01", periods=300, freq="D"))

        for step in range(chain_len):
            amount = round(amount * rng.uniform(0.85, 0.98), 0)
            new_rows.append({
                "txn_id": f"TXN_LAY_{len(new_rows):06d}",
                "sender_id": chain[step],
                "receiver_id": chain[step + 1],
                "amount": amount,
                "date": base_date + pd.Timedelta(hours=int(rng.integers(1, 48))),
                "hour": int(rng.integers(0, 24)),
                "txn_type": "openbank",
                "aml_label": 1,
            })

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


if __name__ == "__main__":
    print("계좌 데이터 생성 중...")
    accounts = generate_accounts()
    print(f"  계좌: {len(accounts):,}개  |  의심 계좌: {accounts['is_suspicious'].sum():,}개")

    print("거래 데이터 생성 중...")
    transactions = generate_transactions(accounts)
    aml = transactions["aml_label"].sum()
    print(f"  거래: {len(transactions):,}건  |  AML: {aml:,}건 ({aml/len(transactions)*100:.1f}%)")

    accounts.to_csv(OUTPUT_DIR / "accounts.csv", index=False, encoding="utf-8-sig")
    transactions.to_csv(OUTPUT_DIR / "transactions.csv", index=False, encoding="utf-8-sig")
    print("저장 완료: accounts.csv, transactions.csv")
