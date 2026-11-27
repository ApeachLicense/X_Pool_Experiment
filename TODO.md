# Third experiment — One-to-Many

GradCache-32 (microbatch 4) and X-Pool text-conditioned pooling remain the baseline.

공통 연구 지침: [AGENTS.md](AGENTS.md). 실행 및 평가 프로토콜: [README.md](README.md#third-experiment-supcon). 현재 활성 과제는 01번이며 02–05번은 별도 요청 후 진행한다.

## 01. Supervised Contrastive Learning (implemented; experiments pending)

NeurIPS 2020 · 이미지 분류 및 표현 학습

동일 video ID를 공유하는 모든 text-video 조합을 positive로 처리한다. SupCon Eq. (2)의 positive별 log-softmax 평균을 X-Pool의 양방향 loss에 적용한다.

- [x] `--loss supcon` 옵션, GradCache 및 일반/누적 학습 경로 연결
- [x] CPU 회귀 테스트 코드 추가 (실행 결과는 별도 기록 필요)
- [ ] CPU 회귀 테스트 실행 및 결과 기록
- [ ] RTX 3060 Ti GPU smoke test (10 updates) 실행 및 결과 기록
- [ ] GradCache-32의 배치 내 동일 영상 ID 중복 빈도 측정
- [ ] Seed 24 본 학습 (9K, 5 epochs, 28,125 updates)
- [ ] Seed 25 본 학습 (9K, 5 epochs, 28,125 updates)
- [ ] 두 seed 최종 체크포인트의 1K-A 평가 및 기존 GradCache-32와 대응 비교
- [ ] 실험 결과·로그·한계 기록 및 README 업데이트
- [ ] 필요시 positive-aware sampler 등 별도 ablation을 내부 검증셋으로 검증 (별도 승인/실험 설계 후 진행)

[논문 읽기](https://arxiv.org/abs/2004.11362)

## 02. Debiased Contrastive Learning

NeurIPS 2020 · 확률통계·표현 학습

Positive와 Negative의 혼합분포를 이용한 손실 함수 편향 보정.

- [ ] 논문 검토 및 X-Pool 적용 설계·구현

[논문 읽기](https://arxiv.org/abs/2007.00224)

## 03. Boosting Contrastive Self-Supervised Learning With False Negative Cancellation

WACV 2022 · 자기지도학습

의미적으로 잘못 분류된 Negative를 탐지하고 처리하는 실용적인 방법.

- [ ] 논문 검토 및 X-Pool 적용 설계·구현

[논문 읽기](https://openaccess.thecvf.com/content/WACV2022/html/Huynh_Boosting_Contrastive_Self-Supervised_Learning_With_False_Negative_Cancellation_WACV_2022_paper.html)

## 04. Bayesian Negative Sampling for Recommendation

ICDE 2023 · 추천 시스템

Bayes' Theorem을 활용한 실제 Negative 확률 추정. 다른 도메인의 방법론을 이식한다는 연구 방향에 부합한다.

- [ ] 논문 검토 및 X-Pool 적용 설계·구현

[논문 읽기](https://www.computer.org/csdl/proceedings-article/icde/2023/222700a749/1PByQ69EYmI)

## 05. Improved Probabilistic Image-Text Representations (PCME++)

ICLR 2024 · 확률적 임베딩

임베딩을 확률분포로 확장하고 적은 연산량으로 확률적 거리를 계산하는 방법.

- [ ] 논문 검토 및 X-Pool 적용 설계·구현

[논문 읽기](https://arxiv.org/abs/2305.18171)
