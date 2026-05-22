# Embedded Guardian — Cross-Chain AML Detection Model

> **임베디드 가디언 특허 기반 크로스체인 자금세탁방지(AML) 탐지 모델**  
> 2계층 하이브리드 아키텍처: 결정론적 규칙 엔진 + GNN(Graph Neural Network)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Private-red.svg)]()
[![F1 Score](https://img.shields.io/badge/F1%20Score-0.942-brightgreen.svg)]()

---

## 개요

본 프로젝트는 **임베디드 가디언(Embedded Guardian)** 특허에 기반한 2계층 하이브리드 AML 탐지 모델의 프로토타입 구현체입니다.

기존 AML 솔루션이 온체인 또는 오프체인 중 하나만 분석하는 것과 달리, 본 모델은 **전통 금융망(이체 그래프)과 퍼블릭 블록체인 데이터를 단일 이종 그래프로 통합**하여 크로스체인 자금세탁 경로를 탐지합니다.

---

## 아키텍처

```
입력 데이터 (이체 트랜잭션 + 계좌 노드 피처)
        │
        ▼
┌─────────────────────────────┐
│  Layer 1: 결정론적 규칙 엔진  │  ← 빠른 1차 필터링
│  R1. 일일 한도 초과 (2천만원) │
│  R2. 비정상 시간대 (새벽 0-5시)│
│  R3. Smurfing 패턴 탐지      │
│  R4. Layering 패턴 탐지      │
│  R5. 신규 계좌 고액 거래      │
└──────────────┬──────────────┘
               │  layer1_score
               ▼
┌─────────────────────────────┐
│  Layer 2: GNN AI 오라클      │  ← 관계 기반 딥러닝
│  GraphSAGE (2-layer)        │
│  노드 피처: 14차원            │
│   - 인구통계 (연령/성별/지역) │
│   - 신용위험/플랫폼리스크     │
│   - 트랜잭션 통계 피처        │
│   - Smurfing 수신 패턴        │
└──────────────┬──────────────┘
               │  layer2_prob
               ▼
┌─────────────────────────────┐
│  앙상블 결합 (L1×0.2 + L2×0.8)│
│  final_flag: AML 의심 계좌   │
└─────────────────────────────┘
```

---

## 성능 지표

합성 데이터(5,000 계좌 / 31,247 거래) 기준 검증 결과:

| 모델 | Precision | Recall | **F1 Score** |
|:---|:---:|:---:|:---:|
| Layer 1 규칙 엔진 | 0.811 | 0.891 | 0.849 |
| Layer 2 GNN | 0.898 | 1.000 | 0.946 |
| **앙상블 최종** | **0.890** | **1.000** | **0.942** |

> 목표 F1 ≥ 0.90 달성

---

## 탐지 대상 AML 패턴

| 패턴 | 설명 | 탐지 방식 |
|:---|:---|:---|
| **Smurfing** | 2천만원 보고 임계치 아래 소액 분산 반복 입금 | Layer 1 R3 + GNN `recv_small_daily_max` 피처 |
| **Layering** | A→B→C→D 연속 이체로 자금 출처 은닉 | Layer 1 R4 + GNN 그래프 엣지 패턴 |
| **Structuring** | 일일 한도를 조금씩 밑도는 반복 거래 | Layer 1 R1 |
| **비정상 시간대** | 새벽 0-5시 집중 거래 | Layer 1 R2 + GNN `dawn_ratio` 피처 |
| **신규 계좌 고액** | 거래 이력 적은 계좌의 갑작스러운 고액 송금 | Layer 1 R5 |

---

## 프로젝트 구조

```
embedded-guardian-aml/
│
├── model/
│   ├── main.py                  # 전체 파이프라인 실행 진입점
│   │
│   ├── data/
│   │   └── synthetic_data.py    # 합성 이체 트랜잭션 데이터 생성기
│   │                            # (NICE평가정보 / BC카드 / NH농협 분포 기반)
│   │
│   ├── eda/
│   │   ├── eda_analysis.py      # 탐색적 데이터 분석 + 시각화
│   │   └── eda_result.png       # AML 패턴 분포 시각화 (6종)
│   │
│   ├── layer1/
│   │   └── rule_engine.py       # 결정론적 규칙 엔진 (R1~R5)
│   │
│   └── layer2/
│       └── gnn_model.py         # GraphSAGE GNN + 앙상블
│
└── .gitignore
```

---

## 설치 및 실행

### 요구 사항

```bash
pip install numpy pandas torch matplotlib scikit-learn networkx
```

### 전체 파이프라인 실행

```bash
cd model
python main.py
```

실행 순서:
1. **Step 1** — 합성 데이터 생성 (`accounts.csv`, `transactions.csv`)
2. **Step 2** — EDA 분석 및 시각화 (`eda_result.png`)
3. **Step 3** — Layer 1 규칙 엔진 실행 (`layer1_results.csv`)
4. **Step 4** — Layer 2 GNN 학습 (`gnn_weights.pt`)
5. **Step 5** — 앙상블 최종 결과 출력 (`ensemble_results.csv`)

### 단계별 개별 실행

```bash
python model/data/synthetic_data.py   # 데이터 생성만
python model/layer1/rule_engine.py    # 규칙 엔진만
python model/layer2/gnn_model.py      # GNN 학습만
```

---

## 실데이터 적용 방법

금융결제원 D-테스트베드 실데이터 사용 시, 아래 형식의 CSV를 `model/data/`에 배치하면 Layer 1~2가 그대로 동작합니다.

**accounts.csv** (계좌 노드)

| 컬럼 | 설명 |
|:---|:---|
| `account_id` | 계좌 일련번호 (가명처리) |
| `gender` | 성별 (1/2) |
| `age_group` | 연령대 |
| `credit_grade` | 신용위험등급 (NICE) |
| `platform_risk` | 플랫폼리스크 ML 스코어 (NICE) |
| `num_accounts` | 자유입출식 계좌 수 (NH농협) |
| `is_suspicious` | AML 라벨 (FF_SP_AI 대리변수) |

**transactions.csv** (이체 엣지)

| 컬럼 | 설명 |
|:---|:---|
| `txn_id` | 거래 고유 ID |
| `sender_id` | 송금 계좌 |
| `receiver_id` | 수신 계좌 |
| `amount` | 거래 금액 (원) |
| `date` | 거래 일자 |
| `hour` | 거래 시각 (0-23) |
| `txn_type` | 거래 유형 (transfer/openbank/atm/cms) |

---

## 기술 스택

| 분류 | 기술 |
|:---|:---|
| 언어 | Python 3.10+ |
| 딥러닝 | PyTorch 2.x (순수 구현 GraphSAGE, torch_geometric 불필요) |
| 그래프 처리 | NetworkX |
| 데이터 처리 | pandas, numpy |
| 시각화 | matplotlib |
| 모델 저장 | `torch.save` (`.pt`) |

---

## 특허 기반

본 프로젝트는 다음 특허의 기술적 구현체입니다:

- **임베디드 가디언 1차 출원** (2026.04.12) — 분산 원장 기반 2계층 임베디드 규제 노드를 이용한 금융 규제 자동화 시스템 및 방법
- **임베디드 가디언 2차 출원** (2026.04.26) — 수탁 검증 계층 및 감독 합의 계층을 포함하는 비대칭 가중 병렬 합의 구조 시스템

---

## 향후 계획

- [ ] 금융결제원 D-테스트베드 실데이터 연동 (OB_TRNS_TRAN, HF_TRNS_TRAN 등 9개 테이블)
- [ ] Bitcoin/Ethereum 퍼블릭 블록체인 주소 클러스터링 통합 (크로스체인 링킹)
- [ ] GAT(Graph Attention Network)로 모델 업그레이드
- [ ] FIU STR 연동 설계서 작성
- [ ] 자금세탁 경로 시각화 대시보드 (Gephi / D3.js)

---

## 연구자

**김한주** — 한양대학교 석사과정  
한국스테이블코인연구소  
`hanjuu@hanyang.ac.kr`

---

> 본 저장소는 비공개 연구 목적으로 관리됩니다. 무단 복제 및 배포를 금지합니다.
