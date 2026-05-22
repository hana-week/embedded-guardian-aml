"""
EDA (탐색적 데이터 분석)
합성 데이터 분포 분석 및 AML 패턴 시각화
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[경고] matplotlib 없음 - 통계 요약만 출력합니다")


def load_data():
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    transactions["date"] = pd.to_datetime(transactions["date"])
    return accounts, transactions


def print_summary(accounts: pd.DataFrame, transactions: pd.DataFrame):
    print("\n" + "=" * 50)
    print("계좌 기본 통계")
    print("=" * 50)
    print(f"총 계좌 수        : {len(accounts):,}")
    print(f"의심 계좌 수      : {accounts['is_suspicious'].sum():,} ({accounts['is_suspicious'].mean()*100:.1f}%)")
    print(f"평균 계좌 보유 수  : {accounts['num_accounts'].mean():.2f}개")
    print(f"평균 신용등급     : {accounts['credit_grade'].mean():.2f}")
    print(f"평균 플랫폼리스크  : {accounts['platform_risk'].mean():.2f}")

    print("\n" + "=" * 50)
    print("거래 기본 통계")
    print("=" * 50)
    print(f"총 거래 건수      : {len(transactions):,}")
    aml = transactions["aml_label"].sum()
    print(f"AML 라벨 건수     : {aml:,} ({aml/len(transactions)*100:.1f}%)")
    print(f"평균 거래 금액    : {transactions['amount'].mean():,.0f}원")
    print(f"중앙값 거래 금액  : {transactions['amount'].median():,.0f}원")
    print(f"최대 거래 금액    : {transactions['amount'].max():,.0f}원")

    print("\n거래 유형 분포:")
    print(transactions["txn_type"].value_counts().to_string())

    dawn = transactions["hour"].isin([0, 1, 2, 3, 4, 5]).mean()
    print(f"\n새벽(00-05시) 거래 비율: {dawn*100:.1f}%")

    print("\n" + "=" * 50)
    print("AML vs 정상 비교")
    print("=" * 50)
    aml_df = transactions[transactions["aml_label"] == 1]
    norm_df = transactions[transactions["aml_label"] == 0]
    print(f"AML 평균 금액   : {aml_df['amount'].mean():>15,.0f}원")
    print(f"정상 평균 금액  : {norm_df['amount'].mean():>15,.0f}원")
    aml_dawn = aml_df["hour"].isin([0,1,2,3,4,5]).mean()
    norm_dawn = norm_df["hour"].isin([0,1,2,3,4,5]).mean()
    print(f"AML 새벽 비율   : {aml_dawn*100:.1f}%")
    print(f"정상 새벽 비율  : {norm_dawn*100:.1f}%")


def account_risk_profile(accounts: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    stats = transactions.groupby("sender_id").agg(
        txn_count=("txn_id", "count"),
        total_sent=("amount", "sum"),
        avg_amount=("amount", "mean"),
        dawn_count=("hour", lambda x: x.isin([0,1,2,3,4,5]).sum()),
    ).reset_index().rename(columns={"sender_id": "account_id"})

    merged = accounts.merge(stats, on="account_id", how="left").fillna(0)
    merged["dawn_ratio"] = (merged["dawn_count"] / merged["txn_count"].replace(0, 1)).round(4)

    print("\n" + "=" * 50)
    print("의심 vs 정상 계좌 행동 차이")
    print("=" * 50)
    for col in ["txn_count", "avg_amount", "dawn_ratio", "num_accounts", "platform_risk"]:
        s = merged[merged["is_suspicious"] == 1][col].mean()
        n = merged[merged["is_suspicious"] == 0][col].mean()
        ratio = s / max(n, 1e-8)
        print(f"{col:20s}  의심: {s:10.2f}  정상: {n:10.2f}  비율: {ratio:.2f}x")

    return merged


def plot_eda(accounts: pd.DataFrame, transactions: pd.DataFrame, merged: pd.DataFrame):
    if not HAS_PLOT:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("AML 탐지 모델 - EDA 결과 (합성 데이터)", fontsize=14, fontweight="bold")

    colors = ["steelblue", "tomato"]
    labels_map = {0: "정상", 1: "의심"}

    # 1. 신용등급 분포
    ax = axes[0, 0]
    for label in [0, 1]:
        grp = accounts[accounts["is_suspicious"] == label]
        ax.hist(grp["credit_grade"], bins=10, alpha=0.6,
                label=labels_map[label], color=colors[label], density=True)
    ax.set_title("신용등급 분포")
    ax.set_xlabel("신용등급 (1=최고위험)")
    ax.legend()

    # 2. 플랫폼리스크 스코어
    ax = axes[0, 1]
    for label in [0, 1]:
        grp = accounts[accounts["is_suspicious"] == label]
        ax.hist(grp["platform_risk"], bins=11, alpha=0.6,
                label=labels_map[label], color=colors[label], density=True)
    ax.set_title("플랫폼리스크 스코어 분포")
    ax.set_xlabel("리스크 등급 (0=저위험)")
    ax.legend()

    # 3. 시간대별 거래 비율
    ax = axes[0, 2]
    for label in [0, 1]:
        grp = transactions[transactions["aml_label"] == label]
        rate = grp.groupby("hour").size() / len(grp)
        ax.plot(rate.index, rate.values, marker="o", markersize=3,
                label="AML" if label else "정상", color=colors[label])
    ax.axvspan(0, 5, alpha=0.1, color="red")
    ax.set_title("시간대별 거래 비율")
    ax.set_xlabel("시간 (시)")
    ax.legend()

    # 4. 거래 금액 (log10)
    ax = axes[1, 0]
    for label in [0, 1]:
        grp = transactions[transactions["aml_label"] == label]
        ax.hist(np.log10(grp["amount"].clip(lower=1)), bins=30, alpha=0.6,
                label="AML" if label else "정상", color=colors[label], density=True)
    ax.set_title("거래 금액 분포 (log10)")
    ax.set_xlabel("log10(금액)")
    ax.legend()

    # 5. 보유 계좌 수
    ax = axes[1, 1]
    pivot = accounts.groupby(["num_accounts", "is_suspicious"]).size().unstack(fill_value=0)
    pivot.plot(kind="bar", ax=ax, color=colors, alpha=0.8)
    ax.set_title("보유 계좌 수 분포")
    ax.set_xlabel("계좌 수")
    ax.legend(["정상", "의심"])
    ax.tick_params(axis="x", rotation=0)

    # 6. 위험 피처 상관관계
    ax = axes[1, 2]
    risk_cols = ["credit_grade", "platform_risk", "num_accounts", "dawn_ratio", "is_suspicious"]
    corr = merged[risk_cols].corr().round(2)
    im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
    tick_labels = ["신용등급", "리스크", "계좌수", "새벽비율", "의심여부"]
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    for i in range(len(risk_cols)):
        for j in range(len(risk_cols)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("위험 피처 상관관계")

    plt.tight_layout()
    out = OUTPUT_DIR / "eda_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n시각화 저장: {out}")


if __name__ == "__main__":
    print("데이터 로딩...")
    accounts, transactions = load_data()
    print_summary(accounts, transactions)
    merged = account_risk_profile(accounts, transactions)
    plot_eda(accounts, transactions, merged)

    merged.to_csv(OUTPUT_DIR / "account_risk_profile.csv", index=False, encoding="utf-8-sig")
    print("\n계좌 위험 프로파일 저장: account_risk_profile.csv")
