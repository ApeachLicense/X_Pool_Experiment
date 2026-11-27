# Third experiment — One-to-Many

GradCache-32 (microbatch 4) and X-Pool text-conditioned pooling remain the baseline.

공통 연구 지침: [AGENTS.md](AGENTS.md). 실행 및 평가 프로토콜: [README.md](README.md#third-experiment-supcon). 현재 활성 과제는 01번이며 02–05번은 별도 요청 후 진행한다.

## 01. Supervised Contrastive Learning — 구현 검수 완료, 두 seed 본 학습 대기

NeurIPS 2020 · 이미지 분류 및 표현 학습

동일 video ID를 공유하는 모든 text-video 조합을 positive로 처리한다. SupCon Eq. (2)의 positive별 log-softmax 평균을 X-Pool의 양방향 loss에 적용한다.

- [x] `--loss supcon` 옵션, GradCache 및 일반/누적 학습 경로 연결
- [x] 회귀 테스트 24개 통과 (`python -m unittest discover -s tests -v`, 종료 코드 0). 고유 ID 배치에서 CLIP 손실·gradient 일치, 중복 ID 배치에서 수식과 일치, 청크 경계를 넘는 positive에서 전체 배치와 GradCache gradient 일치를 확인했다.
- [x] RTX 3060 Ti에서 seed 24·25 각각 10 updates 완료. 두 실행 모두 batch 32, chunk 4, `--loss supcon --check-recompute`에서 유한한 손실, feature 재계산 검사 통과, 체크포인트 저장, 종료 코드 0을 확인했다.
- [x] 9K 학습 데이터와 기존 sampler로 seed별 5 epoch의 배치 순서를 재생해 추가 positive 빈도를 집계했다.

**검수 기록 (2026-11-27)** · [검수한 코드 `3f64b19`](https://github.com/ApeachLicense/Implementing_X-Pool_on_RTX3060Ti/commit/3f64b190799e0ce6430b05fa7dac05796f22e1be). 아래 5 epoch 수치는 영상 ID 집계이고, 실제 GPU 학습 범위는 seed당 10 updates다.

| Seed | GPU 검증 중 중복 ID 배치 | 5 epoch 중 중복 ID 배치 | 추가 positive를 갖는 학습 쌍 |
| --- | ---: | ---: | ---: |
| 24 | 0 / 10 | 1,432 / 28,125 (5.09%) | 2,915 / 900,000 (0.324%) |
| 25 | 1 / 10 | 1,442 / 28,125 (5.13%) | 2,937 / 900,000 (0.326%) |

Seed 25의 10번째 배치에서 `video6853`의 두 캡션이 서로 다른 청크에 들어가 실제 다중 positive 경로를 검증했다. 로컬 검수 기록은 `outputs/supcon_review_s24/`, `outputs/supcon_review_s25/`의 `monitor_report.json`·`training_steps.csv`와 `outputs/supcon_sampler_review.json`에 있다.

**다음 실험은 TODO 01의 두 seed 본 학습이다.** 검수한 설정에서는 위 세 검증을 반복할 필요가 없다. 약 95%의 배치에서 기존 CLIP 손실이 그대로 적용되므로, 현재 sampler를 유지한 상태에서 이 드문 추가 positive가 검색 성능에 미치는 영향을 측정한다. [학습 명령과 기존 대조군](README.md#third-experiment-supcon)

- [ ] 실행 전 동일 seed의 기존 GradCache-32 CLIP 대조군과 설정을 대조하고, 주 지표인 text-to-video R@1의 개선 판정 기준을 기록한다.
- [ ] Seed 24 본 학습 (`supcon32_s24`, 9K, 5 epochs, 28,125 updates)
- [ ] Seed 25 본 학습 (`supcon32_s25`, 동일 조건). 한 GPU에서 두 run을 순차 실행한다.
- [ ] 각 seed의 `global_step=28125`, `partial_epoch=false` 확인 후 최종 체크포인트를 1K-A에서 평가한다. 동일 seed 대조군 대비 R@1·5·10 차이와 두 seed 평균 차이(%p)를 기록한다.
- [ ] 실험 결과·로그·한계 기록 및 README 업데이트
- [ ] 필요시 positive-aware sampler 등 별도 ablation을 내부 검증셋으로 검증 (별도 승인/실험 설계 후 진행)

**검수에서 확인한 주의점**

- 각 본 학습은 새 실험 이름과 새 초기화로 시작하고, 이후 해당 run의 `checkpoint-last.pth`에서만 재개한다. 구형 CLIP 체크포인트에 loss 설정과 동반 프로필이 모두 없으면 현재 재개 검사가 SupCon으로의 전환을 허용한다.
- 중복 ID 배치 집계에는 `duplicate_video_batches`를 사용한다. `duplicate_negative_batches`는 SupCon이 positive로 처리한 동일 ID 쌍도 집계하므로 남은 false negative의 지표로 해석할 수 없다.

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
