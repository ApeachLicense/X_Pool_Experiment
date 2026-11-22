# X-Pool 실험 진행 보고서

2026년 11월 22일 기준 · `second-experiment` · MSR-VTT text-to-video retrieval

RTX 3060 Ti 8GiB 한 장에서 **GradCache-32와 누적-32 모두 5 epochs를 완료**했다. 같은 seed 24의 최종 체크포인트에서 GradCache-32의 **R@1은 45.4%, 누적-32는 43.1%**였다. 데이터 순서와 optimizer 업데이트 스케줄을 맞춘 비교에서 GradCache-32가 **2.3%p** 앞섰다. 현재 확인한 범위는 seed 24 한 쌍이다.

## 출발점과 사전 확인

- 공식 X-Pool 9K 가중치의 MSR-VTT 1K-A R@1 **46.9%**를 재현했다. 기존 배치 4 자체 학습의 최종 5 epoch R@1은 **39.9%**였고, 전체 학습은 약 21시간 55분 걸렸다. [공식 평가](results/msrvtt9k/evaluation_report.json) · [배치 4 학습 기록](results/msrvtt9k_split_training/completion_summary.json)
- 기존 배치 4의 epoch 1–5와 6시간 중간 체크포인트를 비교했다. 1K-A에서 가장 높은 자체 점수는 epoch 4의 **40.3%**였다. 공식 모델만 top-1을 맞힌 117개 질의 중 100개는 자체 모델에서 정답이 2–5위였다. 이 비교를 통해 비슷한 상위 후보를 구분하는 문제가 드러났다. [체크포인트 비교](results/checkpoint_analysis_20261109/metrics_summary.csv) · [실패 사례](results/checkpoint_analysis_20261109/casebook.md)
- Positive cosine margin 0.02는 별도 8K 학습·1K 내부 검증에서 2개 seed로 시험했다. 3,000 updates에서 원래 loss의 평균 R@1은 **40.40%**, margin은 **39.65%**였다. 두 seed 모두 낮아 장시간 실행으로 확대하지 않았다. [Margin 파일럿](results/margin_pilot_20261109/README.md)
- 첫 GradCache 검증에서는 배치 4의 전체 대조 gradient와 2+2 재인코딩 gradient를 비교했다. TF32를 끈 재검증에서 상대 L2 오차는 **2.4e-6–5.6e-6**이었다. 독립 loss를 계산하는 단순 accumulation의 이전 상대 오차는 **0.84–1.04**였다. 이 수치는 실제 GPU에서 GradCache를 본 학습에 적용하기 전에 검증한 gradient 결과다. [Gradient 검증](results/first_experiment_20261115/README.md)
- 배치 4의 feature drift는 학습 단계와 학습률에 따라 달랐다. 오차 비율 0.1 이내의 관측 뱅크 크기는 시작 직후 최대 LR에서 40개, epoch 1에서 800개, epoch 3에서 2,400개 이상이었다. Warmup 끝의 drift는 미측정이라 배치 4 + XBM은 후속 후보로 남겨뒀다. [Drift 측정](results/first_experiment_20261115/README.md)

## Seed 24 대조 실험 설계

Proposal 피드백은 contrastive learning에서 optimizer 업데이트 주기와 **한 loss에 들어가는 in-batch negative 범위**를 구분해야 한다는 점을 짚었다. 이를 확인하기 위해 32쌍이 함께 optimizer에 반영되는 주기를 맞추고, 각 loss를 구성하는 후보 집합을 바꿨다.

| 조건 | 한 optimizer 업데이트 | 한 loss의 후보 | 질의당 negative 슬롯 |
| --- | --- | ---: | ---: |
| GradCache-32 | 32쌍을 4쌍씩 재인코딩하고 전체 32×32 pooling·loss의 gradient로 한 번 업데이트 | 32 | 31 |
| 누적-32 | 4쌍의 독립 loss를 8번 계산하고 예제 수로 가중 평균한 gradient로 한 번 업데이트 | 4 | 3 |

기본 CLIP ViT-B/32와 같은 pooling 초기화, seed 24, MSR-VTT 9K, FP32 입출력과 가중치, 12프레임, 입력 224×224를 사용했다. 본 학습에서는 두 조건 모두 matmul과 cuDNN의 TF32 허용 플래그가 켜져 있었다. CLIP 학습률 1e-6, pooling 학습률 3e-5, warmup 10%, cosine 스케줄과 동일한 optimizer 설정을 썼다. 한 epoch는 180,000 caption-video pairs 및 5,625 updates이고, 5 epochs는 **900,000 pair 노출과 28,125 updates**에 해당한다.

남겨둔 체크포인트에서 두 조건의 **5개 epoch 셔플 순열이 모두 일치**했고, 학습 CSV에서 **28,125 updates의 두 학습률 값**이 모두 일치했다. 동일 영상의 캡션이 한 32쌍 그룹에 중복된 업데이트는 두 조건에서 각각 1,432회였다. 실제 loss 그룹 안에 동일 영상이 중복된 업데이트는 GradCache-32에서 1,432회, 누적-32에서 142회였다. 이 대비로 더 넓은 대조 집합을 쓰는 효과를 비교할 수 있다. 슬롯 수와 함께 같은 영상 캡션의 대조 빈도와 pooling dropout 적용 단위도 달라진다. [실험 계획과 실행법](SECOND_EXPERIMENT.md)

## 실행 경과와 GPU 사용량

11월 19일 RTX 3060 Ti에서 GradCache chunk 2와 4를 각각 10 updates로 확인했다. 첫 두 업데이트를 제외한 평균은 chunk 2가 3.92초, chunk 4가 3.00초였으며 두 설정 모두 encoder 재계산 검사를 통과했다. Chunk 4로 본 학습을 진행했다. 짧은 재개 확인에서는 GradCache chunk 4가 step 10→11, 누적-32가 step 2→3으로 이어졌고 CPU 테스트 36개도 통과했다. [GPU smoke 검증](SECOND_EXPERIMENT.md#windows에서-smoke-실행)

| 실행 | 한국 시간 종료 | 시작 → 종료 update | 누적 epoch | 프로세스 시간 | 장치 VRAM 최대 |
| --- | --- | ---: | --- | ---: | ---: |
| GradCache-32 첫 세션 | 11월 19일 21:22 | 0 → 11,250 | 2 완료 | 8시간 36분 | 7,393 MiB |
| GradCache-32 두 번째 세션 | 11월 20일 22:37 | 11,250 → 22,500 | 4 완료 | 8시간 21분 | 6,754 MiB |
| GradCache-32 마지막 세션 | 11월 21일 12:21 | 22,500 → 28,125 | 5 완료 | 4시간 21분 | 6,979 MiB |
| 누적-32 첫 세션 | 11월 21일 20:55 | 0 → 11,957 | 2 완료 + epoch 3의 707 updates | 8시간 31분 | 6,751 MiB |
| 누적-32 마지막 세션 | 11월 22일 19:25 | 11,957 → 28,125 | 5 완료 | 11시간 35분 | 6,843 MiB |

모든 세션이 정상 종료됐다. 스텝 로그는 두 조건 모두 1–28,125까지 연속이고 모든 기록의 loss 값이 유한했다. 메모리 오류와 GPU 수집 오류는 관측되지 않았다. 각 epoch 체크포인트의 global step, sampler 위치, optimizer step, scheduler step이 일치했다. 최종 평가에는 두 모델 모두 `partial_epoch=false`인 epoch 5 체크포인트를 사용했다.

프로세스 시간을 합하면 GradCache-32는 **21시간 18분 8초**, 누적-32는 **20시간 6분 27초**였다. 이 합계는 모델 로딩·재개·저장을 포함하며 세션 사이 대기와 평가 시간은 제외한다. 이번 측정에서 GradCache-32가 약 **1시간 12분(5.9%)** 더 걸렸다. PyTorch 최대 allocated는 각각 **4,435.4 / 4,431.1 MiB**, 최대 reserved는 각각 **4,796 / 4,804 MiB**였다. 위 표의 장치 수치는 데스크톱 등 다른 프로세스를 포함해 샘플링한 관측 최대값이다.

## MSR-VTT 1K-A 최종 평가

자체 배치 32 두 조건은 학습 중 1K-A 평가를 끈 상태로 실행했다. 각각 고정된 5번째 epoch 체크포인트를 동일한 1K-A에서 한 번 평가했다. 배치 4와 공식 가중치 수치는 이전에 측정한 참고 결과다.

| 조건 | R@1 | R@5 | R@10 | MedR | MeanR |
| --- | ---: | ---: | ---: | ---: | ---: |
| GradCache-32, seed 24 | **45.4%** | **72.5%** | **81.6%** | 2.0 | 14.040 |
| 누적-32, seed 24 | 43.1% | 70.4% | 80.7% | 2.0 | 13.793 |
| 기존 자체 배치 4, epoch 5 | 39.9% | 67.7% | 77.7% | 2.0 | 14.002 |
| 공식 X-Pool 9K 제공 가중치 | 46.9% | 72.8% | 82.2% | 2.0 | 14.291 |

Seed 24에서 GradCache-32는 누적-32보다 **R@1 +2.3%p, R@5 +2.1%p, R@10 +0.9%p**였다. MedR은 같고, 평균 순위는 누적-32가 소폭 낮았다. GradCache-32는 이전 자체 배치 4보다 R@1이 **5.5%p 높았다**. 배치 4와는 업데이트 수와 학습률 스케줄도 달랐고 이전 실행 중 셔플 방식 변경이 있었다. 공식 가중치의 구체적인 모델 선택 과정은 확인하지 못했으므로 46.9%는 재현한 공식 참고값으로 둔다.

## 다음 판정에 필요한 결과

Seed 25에서 **GradCache-32가 누적-32보다 R@1이 높은 방향이 재현되는지** 확인한다. 두 seed의 차이가 각각 양수이고 평균이 **+1.0%p 이상**이면 사전에 정한 기준을 통과한다. 예를 들어 seed 25에서 +0.3%p이면 seed 24의 +2.3%p와 평균이 +1.3%p로 통과한다. Seed 25 차이가 0.0%p이면 양수 조건을 충족하지 않는다. 결과에는 seed별 실제 차이와 평균을 함께 기록한다. [사전 판정 기준](SECOND_EXPERIMENT.md#비교-설계)

- **학습.** 학습 로직은 seed 24와 동일하고, 장시간 재개용 실행기 변경은 커밋 `759b9c5`에 기록했다. 두 조건 모두 기존 CLIP 초기화, seed 25, effective batch 32, chunk 4, matmul·cuDNN TF32 플래그 켜짐, `evals_per_epoch=0`을 유지한다. `gradcache32_full_s25`와 `accum32_full_s25`를 각각 새로 시작한다.
- **첫 10시간 세션.** GradCache-32에 `--hours 9.8`(9시간 48분)을 배정하고 나머지 시간은 시작 준비와 저장에 남긴다. Seed 24 속도 기준으로 2 epochs 완료 후 epoch 3에서 약 1,500–1,600 updates를 더 진행할 것으로 예상한다. 종료 시 저장한 `checkpoint-last.pth`로 이어가며, 이후 GradCache-32는 약 11시간 반, 누적-32는 약 20시간의 학습이 남는다. Seed 24의 질의별 비교는 두 최종 모델의 추론에 약 4–6분이 걸릴 것으로 예상하며, 당일 여유 시간에 진행할 수 있다.
- **완료 확인과 평가.** 두 조건이 각각 28,125 updates를 완료하면 5개 epoch의 셔플 순열과 전체 업데이트의 CLIP·pooling 학습률 기록이 일치하는지 대조한다. 두 학습이 모두 끝난 뒤 각 epoch 5 최종 체크포인트를 1K-A에서 한 번씩 평가한다.

전체 배치 변경의 효과까지 분리하려면 현재 코드로 조건을 맞춘 배치 4 실행도 필요하다. 새 학습 설정을 고르는 실험은 별도 내부 검증셋에서 수행한다.

## 최종 결과의 로컬 근거

아래 경로는 실험을 실행한 PC에서 이 저장소 루트를 기준으로 한 위치다. 세션별 `monitor_report.json`, `training_profile.json`, `training_steps.csv`와 체크포인트는 로컬 `outputs/`에 있다.

| 결과 | 평가 기록 | 체크포인트 SHA-256 |
| --- | --- | --- |
| GradCache-32 epoch 5 | `outputs/sequence_20261121_s24/gradcache_final_evaluation.json` | `aa341c4d7b9dbcdf19446f0f6a7a9e57b23c2332aee2e8e70f4b4b2220786697` |
| 누적-32 epoch 5 | `outputs/sequence_20261122_s24/accum32_final_evaluation.json` | `0d45d954fe8b5938d16ea98e1da758aa4cba556ee4959991fc39b40fa3ae45f5` |

전체 실행 상태와 seed 24 차이는 `outputs/sequence_20261122_s24/workflow.json`에 기록했다. 학습 코드는 브랜치 커밋 `b7bd418`을 기반으로 했으며, 실제 장시간 재개에는 실행기의 시간 제한을 0으로 둘 수 있게 한 로컬 변경과 로컬 실행 스크립트를 사용했다.
