"""
임베디드 가디언 크로스체인 AML 파이프라인
전체 실행 진입점

실행 순서:
  Step 1. 합성 데이터 생성
  Step 2. EDA 분석
  Step 3. Layer 1 규칙 엔진
  Step 4. Layer 2 GNN 학습
  Step 5. 최종 앙상블 결과 출력
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def separator(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def step1_generate_data():
    separator("Step 1 — 합성 데이터 생성")
    from data.synthetic_data import generate_accounts, generate_transactions, OUTPUT_DIR

    accounts = generate_accounts()
    print(f"계좌: {len(accounts):,}개  |  의심: {accounts['is_suspicious'].sum():,}개")

    transactions = generate_transactions(accounts)
    aml = transactions["aml_label"].sum()
    print(f"거래: {len(transactions):,}건  |  AML: {aml:,}건 ({aml/len(transactions)*100:.1f}%)")

    accounts.to_csv(OUTPUT_DIR / "accounts.csv", index=False, encoding="utf-8-sig")
    transactions.to_csv(OUTPUT_DIR / "transactions.csv", index=False, encoding="utf-8-sig")
    print("저장 완료")
    return accounts, transactions


def step2_eda(accounts, transactions):
    separator("Step 2 — EDA 분석")
    from eda.eda_analysis import print_summary, account_risk_profile, plot_eda, OUTPUT_DIR

    print_summary(accounts, transactions)
    merged = account_risk_profile(accounts, transactions)
    plot_eda(accounts, transactions, merged)
    merged.to_csv(OUTPUT_DIR / "account_risk_profile.csv", index=False, encoding="utf-8-sig")
    return merged


def step3_layer1(accounts, transactions):
    separator("Step 3 — Layer 1 규칙 엔진")
    import pandas as pd
    from layer1.rule_engine import RuleEngine, OUTPUT_DIR

    engine = RuleEngine(transactions, accounts)
    results = engine.run_all(verbose=True)

    true_labels = accounts[["account_id", "is_suspicious"]]
    eval_df = results.merge(true_labels, on="account_id")
    tp = int(((eval_df["layer1_flagged"] == 1) & (eval_df["is_suspicious"] == 1)).sum())
    fp = int(((eval_df["layer1_flagged"] == 1) & (eval_df["is_suspicious"] == 0)).sum())
    fn = int(((eval_df["layer1_flagged"] == 0) & (eval_df["is_suspicious"] == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    print(f"Layer 1  P: {p:.3f}  R: {r:.3f}  F1: {f1:.3f}")

    results.to_csv(OUTPUT_DIR / "layer1_results.csv", index=False, encoding="utf-8-sig")
    return results


def step4_layer2(accounts, transactions):
    separator("Step 4 — Layer 2 GNN 학습")
    import torch
    from layer2.gnn_model import (build_graph, GNN_AML, train, evaluate,
                                   ensemble_score, EPOCHS, LR, SEED,
                                   DEVICE, OUTPUT_DIR)
    import pandas as pd

    torch.manual_seed(SEED)
    x, adj, labels, mask = build_graph(accounts, transactions)

    model = GNN_AML(in_dim=x.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    n_pos = labels.sum().item()
    pos_weight = (len(labels) - n_pos) / max(n_pos, 1)

    best_f1, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, x, adj, labels, mask, optimizer, pos_weight)
        scheduler.step()
        if epoch % 20 == 0:
            m = evaluate(model, x, adj, labels, mask)
            print(f"  Epoch {epoch:3d} | Loss {loss:.4f} | F1 {m['f1']:.3f}")
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    final = evaluate(model, x, adj, labels, mask)
    print(f"Layer 2  P: {final['precision']:.3f}  R: {final['recall']:.3f}  F1: {final['f1']:.3f}")

    torch.save(model.state_dict(), OUTPUT_DIR / "gnn_weights.pt")

    layer1_path = ROOT / "layer1" / "layer1_results.csv"
    if layer1_path.exists():
        layer1_df = pd.read_csv(layer1_path)
        ensemble = ensemble_score(layer1_df, final["prob"], accounts)
        ensemble.to_csv(OUTPUT_DIR / "ensemble_results.csv", index=False, encoding="utf-8-sig")

        tp = int(((ensemble["final_flag"] == 1) & (ensemble["is_suspicious"] == 1)).sum())
        fp = int(((ensemble["final_flag"] == 1) & (ensemble["is_suspicious"] == 0)).sum())
        fn = int(((ensemble["final_flag"] == 0) & (ensemble["is_suspicious"] == 1)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1_e = 2 * p * r / max(p + r, 1e-8)

        separator("Step 5 — 앙상블 최종 결과")
        print(f"Precision : {p:.3f}")
        print(f"Recall    : {r:.3f}")
        print(f"F1 Score  : {f1_e:.3f}  (목표: 0.90)")
        flagged = ensemble["final_flag"].sum()
        print(f"최종 플래그 계좌: {flagged:,}개 / {len(ensemble):,}개")
        print("\n저장 완료: ensemble_results.csv, gnn_weights.pt")


if __name__ == "__main__":
    t0 = time.time()
    accounts, transactions = step1_generate_data()
    step2_eda(accounts, transactions)
    step3_layer1(accounts, transactions)
    step4_layer2(accounts, transactions)
    print(f"\n전체 실행 완료: {time.time()-t0:.1f}초")
