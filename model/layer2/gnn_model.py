"""
Layer 2 — GNN (Graph Neural Network) AML 탐지 모델
임베디드 가디언 2계층 아키텍처의 2계층: AI 오라클

구조:
  - GraphSAGE (Mean Aggregation) - 순수 PyTorch 구현
  - 입력: 계좌 노드 피처 + 이체 엣지
  - 출력: 계좌별 AML 확률 (0~1)
  - 학습: Proxy 라벨 기반 준지도 학습
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
LAYER1_DIR = Path(__file__).parent.parent / "layer1"
OUTPUT_DIR = Path(__file__).parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 64
OUTPUT_DIM = 1
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 200
LR = 0.005
SEED = 42


# ── 그래프 구성 ────────────────────────────────────────

def build_graph(accounts: pd.DataFrame, transactions: pd.DataFrame):
    """
    계좌 노드 피처 텐서와 정규화된 인접 행렬 반환.
    Returns:
        x      : (N, F) 노드 피처
        adj    : (N, N) 정규화 인접 행렬 (희소하지 않은 dense — 소규모용)
        labels : (N,)  0/1 라벨
        mask   : (N,)  라벨 있는 노드 마스크
    """
    torch.manual_seed(SEED)
    n = len(accounts)
    acc_idx = {acc_id: i for i, acc_id in enumerate(accounts["account_id"])}

    # ── 트랜잭션 파생 피처 계산 ──
    txn = transactions.copy()
    txn["date"] = pd.to_datetime(txn["date"])
    dawn_hours = {0, 1, 2, 3, 4, 5}

    sent = txn.groupby("sender_id").agg(
        txn_count=("txn_id", "count"),
        avg_amount=("amount", "mean"),
        total_sent=("amount", "sum"),
        dawn_sent=("hour", lambda x: x.isin(dawn_hours).sum()),
        openbank_sent=("txn_type", lambda x: (x == "openbank").sum()),
    ).reset_index().rename(columns={"sender_id": "account_id"})
    sent["dawn_ratio"] = sent["dawn_sent"] / sent["txn_count"].clip(lower=1)
    sent["openbank_ratio"] = sent["openbank_sent"] / sent["txn_count"].clip(lower=1)
    sent["avg_amount_log"] = np.log1p(sent["avg_amount"])
    sent["total_sent_log"] = np.log1p(sent["total_sent"])

    recv = txn.groupby("receiver_id").agg(
        recv_count=("txn_id", "count"),
    ).reset_index().rename(columns={"receiver_id": "account_id"})

    # 소액 분산 입금 최대 일별 건수 (smurfing 핵심 피처)
    small_txn = txn[txn["amount"] < 19_800_000].copy()
    small_txn["txn_date"] = small_txn["date"].dt.date
    small_daily = (
        small_txn.groupby(["receiver_id", "txn_date"])
        .size()
        .reset_index(name="daily_count")
        .groupby("receiver_id")["daily_count"]
        .max()
        .reset_index()
        .rename(columns={"receiver_id": "account_id", "daily_count": "recv_small_daily_max"})
    )
    recv = recv.merge(small_daily, on="account_id", how="left").fillna(0)

    feats_df = accounts[["account_id", "gender", "age_group", "credit_grade",
                          "platform_risk", "num_accounts",
                          "dawn_txn_ratio", "easy_pay_cnt"]].copy().astype(
        {"gender": float, "age_group": float, "credit_grade": float,
         "platform_risk": float, "num_accounts": float,
         "dawn_txn_ratio": float, "easy_pay_cnt": float}
    )
    feats_df = feats_df.merge(
        sent[["account_id", "txn_count", "dawn_ratio", "openbank_ratio",
              "avg_amount_log", "total_sent_log"]],
        on="account_id", how="left"
    ).merge(
        recv[["account_id", "recv_count", "recv_small_daily_max"]],
        on="account_id", how="left"
    ).fillna(0)

    feat_cols = ["gender", "age_group", "credit_grade", "platform_risk",
                 "num_accounts", "dawn_txn_ratio", "easy_pay_cnt",
                 "txn_count", "dawn_ratio", "openbank_ratio",
                 "avg_amount_log", "total_sent_log",
                 "recv_count", "recv_small_daily_max"]
    feats = feats_df[feat_cols].copy().astype(float)

    # min-max 정규화
    feats = (feats - feats.min()) / (feats.max() - feats.min() + 1e-8)

    x = torch.tensor(feats.values, dtype=torch.float32)

    # ── 인접 행렬 구성 ──
    # 메모리 효율: 거래 건수가 많으면 상위 N건만 사용
    max_edges = 20_000
    if len(transactions) > max_edges:
        txn_sample = transactions.sample(max_edges, random_state=SEED)
    else:
        txn_sample = transactions

    adj = torch.zeros(n, n, dtype=torch.float32)
    valid_count = 0
    for _, row in txn_sample.iterrows():
        s = acc_idx.get(row["sender_id"])
        r = acc_idx.get(row["receiver_id"])
        if s is not None and r is not None and s != r:
            weight = min(float(row["amount"]) / 1e8, 1.0)  # 금액 기반 엣지 가중치
            adj[s, r] += weight
            valid_count += 1

    # 행 정규화 (D^-1 * A)
    row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
    adj_norm = adj / row_sum

    # 자기 연결 추가 (self-loop)
    adj_norm = adj_norm + torch.eye(n)

    # ── 라벨 및 마스크 ──
    labels = torch.tensor(accounts["is_suspicious"].values, dtype=torch.float32)

    # 라벨 불균형 보정: 의심 계좌 비율이 낮으므로 전체를 학습에 사용
    mask = torch.ones(n, dtype=torch.bool)

    print(f"그래프 구성 완료: {n}개 노드 | {valid_count}개 유효 엣지 | "
          f"피처 차원: {x.shape[1]} | 의심 계좌: {int(labels.sum())}개")

    return x.to(DEVICE), adj_norm.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE)


# ── 모델 정의 ──────────────────────────────────────────

class SAGEConv(nn.Module):
    """GraphSAGE Mean Aggregation 레이어 (순수 PyTorch)"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # 자신 + 이웃 평균을 concat → linear
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # 이웃 평균 집계: (N, N) x (N, F) -> (N, F)
        neighbor_agg = torch.mm(adj, x)
        # 자신과 이웃 concat
        h = torch.cat([x, neighbor_agg], dim=1)
        h = self.linear(h)
        h = self.bn(h)
        return F.relu(h)


class GNN_AML(nn.Module):
    """2-layer GraphSAGE AML 분류기"""

    def __init__(self, in_dim: int, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, OUTPUT_DIM)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = conv(h, adj)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return torch.sigmoid(self.classifier(h)).squeeze(-1)


# ── 학습 루프 ──────────────────────────────────────────

def train(model, x, adj, labels, mask, optimizer, pos_weight):
    model.train()
    optimizer.zero_grad()
    pred = model(x, adj)
    # 불균형 데이터용 가중 BCE
    loss = F.binary_cross_entropy(
        pred[mask], labels[mask],
        weight=(labels[mask] * (pos_weight - 1) + 1)
    )
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, x, adj, labels, mask):
    model.eval()
    pred = model(x, adj)
    prob = pred[mask].cpu().numpy()
    true = labels[mask].cpu().numpy().astype(int)
    binary = (prob >= 0.5).astype(int)

    tp = int(((binary == 1) & (true == 1)).sum())
    fp = int(((binary == 1) & (true == 0)).sum())
    fn = int(((binary == 0) & (true == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "prob": prob}


# ── Layer 1 + Layer 2 앙상블 ──────────────────────────

def ensemble_score(layer1_df: pd.DataFrame, layer2_prob: np.ndarray,
                   accounts: pd.DataFrame, w1=0.4, w2=0.6) -> pd.DataFrame:
    """두 계층 점수를 가중 합산"""
    result = accounts[["account_id", "is_suspicious"]].copy()
    result = result.merge(
        layer1_df[["account_id", "layer1_score", "layer1_flagged", "triggered_rules"]],
        on="account_id", how="left"
    )
    result["layer1_score"] = result["layer1_score"].fillna(0.0)
    result["layer2_prob"] = layer2_prob
    result["ensemble_score"] = (w1 * result["layer1_score"] + w2 * result["layer2_prob"]).round(4)
    # GNN을 주 결정자로 사용 (풍부한 피처로 정밀도 우세)
    # Layer1 score가 높고 GNN이 어느 정도 동의하면 함께 플래그 (2계층 시너지)
    result["final_flag"] = (
        (result["layer2_prob"] >= 0.5) |
        ((result["layer1_flagged"] == 1) & (result["layer2_prob"] >= 0.3))
    ).astype(int)
    return result


# ── 메인 실행 ──────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(SEED)
    print(f"디바이스: {DEVICE}\n")

    print("데이터 로딩...")
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")

    x, adj, labels, mask = build_graph(accounts, transactions)

    in_dim = x.shape[1]
    model = GNN_AML(in_dim=in_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 클래스 불균형 보정 가중치
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f"양성 가중치: {pos_weight:.2f} (불균형 보정)\n")

    print("학습 시작...")
    best_f1 = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, x, adj, labels, mask, optimizer, pos_weight)
        scheduler.step()

        if epoch % 10 == 0:
            metrics = evaluate(model, x, adj, labels, mask)
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | "
                  f"F1: {metrics['f1']:.3f}")
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 최적 모델 복원
    if best_state:
        model.load_state_dict(best_state)

    final_metrics = evaluate(model, x, adj, labels, mask)
    print(f"\n── Layer 2 최종 성능 ──")
    print(f"Precision : {final_metrics['precision']:.3f}")
    print(f"Recall    : {final_metrics['recall']:.3f}")
    print(f"F1 Score  : {final_metrics['f1']:.3f}")

    # Layer 1 + Layer 2 앙상블
    layer1_path = LAYER1_DIR / "layer1_results.csv"
    if layer1_path.exists():
        layer1_df = pd.read_csv(layer1_path)
        ensemble = ensemble_score(layer1_df, final_metrics["prob"], accounts)

        tp = int(((ensemble["final_flag"] == 1) & (ensemble["is_suspicious"] == 1)).sum())
        fp = int(((ensemble["final_flag"] == 1) & (ensemble["is_suspicious"] == 0)).sum())
        fn = int(((ensemble["final_flag"] == 0) & (ensemble["is_suspicious"] == 1)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1_ens = 2 * p * r / max(p + r, 1e-8)

        print(f"\n── 앙상블 (Layer1 x0.4 + Layer2 x0.6) ──")
        print(f"Precision : {p:.3f}")
        print(f"Recall    : {r:.3f}")
        print(f"F1 Score  : {f1_ens:.3f}")

        ensemble.to_csv(OUTPUT_DIR / "ensemble_results.csv", index=False, encoding="utf-8-sig")
        print("\n저장 완료: ensemble_results.csv")

    torch.save(model.state_dict(), OUTPUT_DIR / "gnn_weights.pt")
    print("모델 가중치 저장 완료: gnn_weights.pt")
