"""
Layer 1 — 결정론적 규칙 엔진
임베디드 가디언 2계층 아키텍처의 1계층: 규칙 기반 AML 탐지

규칙 목록:
  R1. 일일 한도 초과 (2천만원)
  R2. 비정상 시간대 거래 (00-05시)
  R3. Smurfing 패턴 (소액 분산 반복 입금)
  R4. Layering 패턴 (단기 연속 이체)
  R5. 신규 계좌 고액 거래
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

THRESHOLD_DAILY = 20_000_000       # 일일 이체 임계치 (원)
THRESHOLD_SMURFING_AMT = 19_800_000  # smurfing 단위 금액 상한
THRESHOLD_SMURFING_COUNT = 3       # smurfing 최소 반복 횟수
THRESHOLD_LAYERING_HOURS = 48      # layering 판정 시간 창 (시간)
THRESHOLD_LAYERING_HOPS = 3        # layering 최소 홉 수
DAWN_HOURS = {0, 1, 2, 3, 4, 5}


@dataclass
class RuleResult:
    rule_id: str
    triggered: bool
    score: float          # 0.0 ~ 1.0
    detail: str = ""


@dataclass
class AccountRisk:
    account_id: str
    total_score: float = 0.0
    triggered_rules: List[str] = field(default_factory=list)
    is_flagged: bool = False   # 총점 0.5 이상 시 플래그


class RuleEngine:
    def __init__(self, transactions: pd.DataFrame, accounts: pd.DataFrame):
        self.txn = transactions.copy()
        self.acc = accounts.copy()
        self.txn["date"] = pd.to_datetime(self.txn["date"])
        self._precompute()

    def _precompute(self):
        """반복 사용하는 집계를 미리 계산"""
        self.daily_send = (
            self.txn.groupby(["sender_id", self.txn["date"].dt.date])["amount"]
            .sum()
            .reset_index()
            .rename(columns={"date": "txn_date", "amount": "daily_amount"})
        )
        self.send_counts = self.txn.groupby("sender_id").size().rename("send_count")
        self.recv_counts = self.txn.groupby("receiver_id").size().rename("recv_count")

    # ── 규칙 정의 ──────────────────────────────────────

    def r1_daily_limit(self, account_id: str) -> RuleResult:
        """R1: 단일 계좌 일일 송금액 2천만원 초과"""
        rows = self.daily_send[self.daily_send["sender_id"] == account_id]
        if rows.empty:
            return RuleResult("R1", False, 0.0)
        max_daily = rows["daily_amount"].max()
        if max_daily > THRESHOLD_DAILY:
            ratio = min(max_daily / THRESHOLD_DAILY, 3.0)
            score = min(0.4 + (ratio - 1) * 0.15, 1.0)
            return RuleResult("R1", True, round(score, 3),
                               f"일일 최대 송금액 {max_daily:,.0f}원 (기준 {THRESHOLD_DAILY:,}원)")
        return RuleResult("R1", False, 0.0)

    def r2_unusual_time(self, account_id: str) -> RuleResult:
        """R2: 새벽(00-05시) 거래 비율 이상"""
        sent = self.txn[self.txn["sender_id"] == account_id]
        if len(sent) < 3:
            return RuleResult("R2", False, 0.0)
        dawn_ratio = sent["hour"].isin(DAWN_HOURS).mean()
        if dawn_ratio > 0.25:
            score = min(dawn_ratio * 1.2, 1.0)
            return RuleResult("R2", True, round(score, 3),
                               f"새벽 거래 비율 {dawn_ratio*100:.1f}%")
        return RuleResult("R2", False, 0.0)

    def r3_smurfing(self, account_id: str) -> RuleResult:
        """R3: 특정 계좌로의 소액 반복 입금 (2천만원 미만 분산)"""
        recv = self.txn[
            (self.txn["receiver_id"] == account_id) &
            (self.txn["amount"] < THRESHOLD_SMURFING_AMT)
        ]
        if len(recv) < THRESHOLD_SMURFING_COUNT:
            return RuleResult("R3", False, 0.0)

        # 동일 날짜 소액 입금 반복 검사
        daily_recv = recv.groupby(recv["date"].dt.date).size()
        max_daily_recv = daily_recv.max()
        if max_daily_recv >= THRESHOLD_SMURFING_COUNT:
            # 건수가 많을수록 높은 점수 (단독으로도 플래그 가능하도록 0.8 기준)
            score = min(0.8 + (max_daily_recv - THRESHOLD_SMURFING_COUNT) * 0.05, 1.0)
            return RuleResult("R3", True, round(score, 3),
                               f"단일일 소액 입금 {max_daily_recv}건 탐지")
        return RuleResult("R3", False, 0.0)

    def r4_layering(self, account_id: str) -> RuleResult:
        """R4: 단기간 내 연속 이체 (받자마자 바로 송금)"""
        recv = self.txn[self.txn["receiver_id"] == account_id][["date", "amount"]].copy()
        send = self.txn[self.txn["sender_id"] == account_id][["date", "amount"]].copy()
        if recv.empty or send.empty:
            return RuleResult("R4", False, 0.0)

        recv = recv.sort_values("date")
        send = send.sort_values("date")

        chain_count = 0
        for _, r_row in recv.iterrows():
            # 입금 후 48시간 이내 송금이 있으면 layering 의심
            window_end = r_row["date"] + pd.Timedelta(hours=THRESHOLD_LAYERING_HOURS)
            rapid_sends = send[(send["date"] > r_row["date"]) & (send["date"] <= window_end)]
            if not rapid_sends.empty:
                chain_count += 1

        if chain_count >= THRESHOLD_LAYERING_HOPS:
            score = min(0.5 + chain_count * 0.05, 1.0)
            return RuleResult("R4", True, round(score, 3),
                               f"단기 연속 이체 체인 {chain_count}건")
        return RuleResult("R4", False, 0.0)

    def r5_new_account_high_value(self, account_id: str) -> RuleResult:
        """R5: 신규 계좌(거래 이력 적음) + 고액 거래"""
        sent = self.txn[self.txn["sender_id"] == account_id]
        if len(sent) == 0:
            return RuleResult("R5", False, 0.0)

        # 총 거래 건수가 적고(신규 계좌 proxy) 평균 금액이 높으면 의심
        if len(sent) <= 5 and sent["amount"].mean() > 10_000_000:
            score = 0.6
            return RuleResult("R5", True, score,
                               f"거래 {len(sent)}건, 평균 {sent['amount'].mean():,.0f}원")
        return RuleResult("R5", False, 0.0)

    # ── 계좌별 종합 평가 ──────────────────────────────

    def evaluate_account(self, account_id: str) -> AccountRisk:
        rules = [
            self.r1_daily_limit,
            self.r2_unusual_time,
            self.r3_smurfing,
            self.r4_layering,
            self.r5_new_account_high_value,
        ]
        # 규칙 가중치 (합계 = 1.0)
        weights = {"R1": 0.25, "R2": 0.15, "R3": 0.25, "R4": 0.25, "R5": 0.10}

        risk = AccountRisk(account_id=account_id)
        for rule_fn in rules:
            result = rule_fn(account_id)
            if result.triggered:
                risk.triggered_rules.append(result.rule_id)
                risk.total_score += result.score * weights[result.rule_id]

        risk.total_score = round(min(risk.total_score, 1.0), 4)
        # 단일 강한 신호(R3/R4 탐지)도 단독 플래그 가능하도록 임계치 낮춤
        risk.is_flagged = risk.total_score >= 0.15
        return risk

    def run_all(self, verbose=True) -> pd.DataFrame:
        """전체 계좌 평가 실행"""
        account_ids = self.acc["account_id"].unique()
        results = []

        for i, acc_id in enumerate(account_ids):
            if verbose and i % 500 == 0:
                print(f"  진행: {i}/{len(account_ids)}")
            risk = self.evaluate_account(acc_id)
            results.append({
                "account_id": risk.account_id,
                "layer1_score": risk.total_score,
                "triggered_rules": ",".join(risk.triggered_rules),
                "layer1_flagged": int(risk.is_flagged),
            })

        df = pd.DataFrame(results)
        flagged = df["layer1_flagged"].sum()
        if verbose:
            print(f"\nLayer 1 결과: {flagged:,}개 계좌 플래그 ({flagged/len(df)*100:.1f}%)")
        return df


if __name__ == "__main__":
    print("데이터 로딩...")
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")

    print("Layer 1 규칙 엔진 실행...")
    engine = RuleEngine(transactions, accounts)
    results = engine.run_all()

    # 실제 라벨과 비교
    true_labels = accounts[["account_id", "is_suspicious"]]
    eval_df = results.merge(true_labels, on="account_id")

    tp = ((eval_df["layer1_flagged"] == 1) & (eval_df["is_suspicious"] == 1)).sum()
    fp = ((eval_df["layer1_flagged"] == 1) & (eval_df["is_suspicious"] == 0)).sum()
    fn = ((eval_df["layer1_flagged"] == 0) & (eval_df["is_suspicious"] == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"\n── Layer 1 성능 ──")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1 Score  : {f1:.3f}")

    results.to_csv(OUTPUT_DIR / "layer1_results.csv", index=False, encoding="utf-8-sig")
    print("\n저장 완료: layer1_results.csv")
