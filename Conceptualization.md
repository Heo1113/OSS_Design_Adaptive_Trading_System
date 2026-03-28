# 1. Conceptualization

**Project Title: OSS_Design_Adaptive-Trading-System** 

### [ Revision history ]

| Revision date | Version # | Description | Author |
| :--- | :--- | :--- | :--- |
| 2026-03-27 | 1.0.0 | Initial Draft  | 허주호 |

---

### 1. Business purpose

> **Project background, motivation, Goal, Target market etc.**

* **Background & Motivation**: 암호화폐 선물 시장은 24시간 운영되며 변동성이 매우 큼. 기존의 정적인 자동 매매 시스템은 급변하는 시장 추세에 유연하게 대응하지 못해 수익성이 저하되는 한계가 있음.
* **Goal**: 바이낸스(Binance) API를 활용해 실시간 데이터를 수집하고, **유전 알고리즘(Genetic Algorithm)**을 통해 매매 파라미터를 스스로 최적화하는 '지능형 적응 매매 엔진'을 구축한다.
* **Target market**: 감정을 배제한 데이터 기반 투자를 원하는 개인 투자자 및 시스템 트레이딩 개발자.

---

### 2. System context diagram

![Use Case Diagram](./images/UseCaseDiagram1.png)



#### [ Description for the terms ]
* **System**: 유전 알고리즘 기반 최적화 엔진과 실시간 매매 실행 로직이 통합된 본 프로젝트의 핵심 시스템.
* **User (Actor)**: API Key, 심볼, 레버리지 등 초기 설정값을 입력하고 시스템의 성과 리포트를 확인하는 주체임.
* **Binance API (Actor)**: 실시간/과거 시세 데이터를 제공하고, 시스템의 주문 명령을 받아 체결 결과를 반환하는 외부 거래소 플랫폼임.
* **TA Library (Actor)**: `pandas_ta` 등 오픈소스 라이브러리를 통해 시세 데이터를 분석하여 지표(ADX, Slope 등)를 제공함.

---

### 3. Use case list

> **Find use cases in your project**

| No | Use Case | Actor | Description |
| :--- | :--- | :--- | :--- |
| 1 | **시세 데이터 수집** | Binance API | 거래소 API를 통해 분석에 필요한 과거 및 실시간 캔들 데이터를 로드함. |
| 2 | **전략 최적화** | System | 유전 알고리즘을 구동하여 현재 시장에 최적화된 파라미터를 탐색함. |
| 3 | **자동 주문 실행** | Binance API | 최적화된 조건 충족 시 거래소 API를 통해 매수/매도 주문을 전송함. |

---

### 4. Concept of operation

> **How to operate the use cases**

#### 1) 전략 최적화 (Strategy Optimization)
* **Purpose**: 시장 상황에 맞는 최적의 매매 파라미터 조합 도출.
* **Approach**: GA 기반 개체군 진화 및 적합도(ROI, PF) 평가를 통해 우수 유전자 선별.
* **Dynamics**: 학습 데이터 구간 종료 시점마다 자동 실행.
* **Goals**: 높은 수익성과 낮은 리스크(MDD)를 가진 전략 파라미터 확보.

#### 2) 자동 매매 실행 (Auto Trading)
* **Purpose**: 기계적 매매를 통한 수익 실현 및 리스크 관리.
* **Approach**: 최적화 엔진이 도출한 임계값과 실시간 지표를 대조하여 즉각적인 API 주문 실행.
* **Dynamics**: 매수/매도 신호가 발생하는 실시간 시점.
* **Goals**: 정확한 타이밍의 체결 및 감정 배제 매매 구현.

---

### 5. Problem statement

> **Technical difficulties and Non-Functional Requirements (NFRs)**

* **연산 효율성**: 유전 알고리즘의 막대한 계산량을 해결하기 위해 `multiprocessing` 기반 병렬 처리 아키텍처를 설계함.
* **데이터 과적합(Overfitting)**: 과거 데이터에만 최적화되는 문제를 방지하기 위해 **Walk-Forward 슬라이딩 윈도우** 기법을 적용함.
* **보안 및 은닉화 (NFRs)**: API Key 보호를 위해 환경 변수(`.env`) 관리 및 객체지향적 변수 은닉화 설계를 도입함.

---

### 6. Glossary

* **GA (Genetic Algorithm)**: 생물의 진화 원리를 모방하여 최적의 해를 찾는 탐색 알고리즘.
* **WFO (Walk-Forward Optimization)**: 학습과 검증 구간을 이동하며 전략의 유효성을 검증하는 기법.
* **GPL v3.0**: 본 프로젝트에 적용된 Strong Copyleft 오픈소스 라이선스.

---

### 7. References

* Binance API Documentation: https://binance-docs.github.io/apidocs/
* Pandas-ta Library: https://github.com/twopirllc/pandas-ta
