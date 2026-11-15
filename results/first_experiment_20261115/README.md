# 긴 학습 전 사전 확인 3종: 실행 결과 (2026-11-15)

[FIRST_EXPERIMENT.md](../../FIRST_EXPERIMENT.md)의 세 명령을 Windows와 RTX 3060 Ti(8GB)에서 [`run_first_experiment.bat`](../../run_first_experiment.bat)으로 차례로 실행했다. 코드는 `ccb0a2d`이고 옵션은 모두 기본값이다(배치 4, worker 2개, seed 24, FP32, 12프레임, 224px). 세 명령 모두 exit code 0으로 끝났고, 모델 로딩까지 포함해 총 **6분 24초**가 걸렸다(steptime 79초, gradcheck 37초, drift 268초).

| 질문 | 측정값 | 판정 |
| --- | --- | --- |
| GradCache gradient가 배치 전체 gradient와 같은가 | 상대 L2 오차 8.1e-4~1.3e-3(허용치 1e-2), 재인코딩 차이 0.0, `forward == pool(encode)` 비트 단위 일치 | **PASSED** |
| 영상 디코딩이 병목인가 | `data_wait` 1.4%, `gpu_step` 93.8%. 로더 공급 16.8 samples/s, GPU 소비 10.6 samples/s | **gpu-bound**: 프레임을 미리 추출할 필요가 없다 |
| 저장한 feature를 몇 step 뒤까지 쓸 수 있는가 | 가장 짧게 잰 lag 50 step(뱅크 200개)에서 이미 오차 비율 0.141 > 0.1 | 휴리스틱 기준 뱅크 크기 **0** |

## 1. gradcheck

| batch | loss | GradCache rel_err | GradCache cos | accumulation rel_err | peak MiB (full / GradCache) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0697 | 1.26e-3 | 0.999999 | 1.04 | 2558 / 2170 |
| 1 | 0.9319 | 1.24e-3 | 0.999999 | 1.00 | 3133 / 2170 |
| 2 | 0.8681 | 8.14e-4 | 1.000000 | 0.84 | 3126 / 2168 |

- 파라미터 그룹별(CLIP text, CLIP vision, pooling, logit scale) 오차도 모두 2e-3 이하다.
- 같은 샘플을 두 번 인코딩한 결과의 차이(`feature_recompute_max_abs_diff`)는 세 batch 모두 0.0이다. 인코더에는 난수가 없다.
- CPU FP32 무작위 가중치 테스트의 1.2e-7보다 오차가 4자리 정도 크다. **원인은 확인하지 않았다.** 가장 유력한 후보는 TF32다. PyTorch 1.8은 Ampere GPU에서 matmul에 TF32를 기본으로 쓰고(`torch.backends.cuda.matmul.allow_tf32=True`), 배치 전체와 chunk는 서로 다른 크기로 계산한다. TF32를 끄고 다시 돌리면 확인할 수 있다.
- 단순 gradient accumulation의 오차는 0.84~1.04로, 예상대로 정답 gradient와 다르다.
- peak는 PyTorch 할당 기준이고 데스크톱이 쓰는 VRAM은 빠져 있다. chunk 2개짜리 GradCache는 batch 1·2에서 배치 4 전체보다 약 960MiB 덜 썼다. batch 0에서는 full의 peak 자체가 낮아서 차이가 390MiB였다.

## 2. steptime

100 step 평균(앞 10 step 제외) 0.383초:

| 구간 | 초 | 비중 |
| --- | ---: | ---: |
| `data_wait` | 0.0053 | 1.4% |
| `tokenize` | 0.0016 | 0.4% |
| `to_device` | 0.0169 | 4.4% |
| `gpu_step` | 0.3591 | 93.8% |

- 프로세스 하나에서 영상 1개를 처리하는 시간은 디코딩 0.112초, 전처리 0.007초다. worker 2개면 초당 16.8개를 공급할 수 있다.
- `data_wait`가 완전히 사라져도 빨라지는 정도는 최대 1.014배다.
- **추정치**: 5 epoch 기준 기존 방식은 23.9시간, GradCache-32는 27.7시간(1.16배)이다. GradCache-32는 update 하나에 chunk 8개를 쓰고, chunk 하나의 no-grad 인코딩에 0.060초가 든다. 이 계산은 배치 32 로더의 부하를 반영하지 않는다. 그래도 GradCache-32의 소비 속도(32 ÷ 3.54초 ≈ 9.0 samples/s)는 로더 공급량보다 작다.

## 3. drift

`--lr-mode peak`(CLIP 1e-6, pooling 3e-5 고정)로 사전학습 CLIP에서 출발해 600 step을 학습했다. 첫 50 step의 평균 loss는 0.48이었고 600 step 전체 평균은 0.28이었다. 오차 비율은 `max(logit_err_video, logit_err_text) ÷ neg_logit_std`이며, 판정 규칙이 이 값을 0.1과 비교한다.

| lag (step) | 뱅크 크기 | frame cos-dist | text cos-dist | logit_err video | logit_err text | neg_logit_std | 오차 비율 | hardest neg 일치 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 200 | 0.0121 | 0.0039 | 0.70 | 0.51 | 4.98 | **0.141** | 0.93 |
| 100 | 400 | 0.0224 | 0.0101 | 1.09 | 0.88 | 5.08 | **0.215** | 0.90 |
| 150 | 600 | 0.0286 | 0.0176 | 1.36 | 1.20 | 5.18 | **0.263** | 0.88 |
| 200 | 800 | 0.0342 | 0.0271 | 1.66 | 1.56 | 5.26 | **0.316** | 0.85 |
| 250 | 1,000 | 0.0358 | 0.0375 | 1.79 | 1.87 | 5.33 | **0.351** | 0.86 |
| 300 | 1,200 | 0.0400 | 0.0487 | 2.07 | 2.28 | 5.36 | **0.426** | 0.86 |
| 350 | 1,400 | 0.0433 | 0.0575 | 2.39 | 2.70 | 5.36 | **0.504** | 0.84 |
| 400 | 1,600 | 0.0473 | 0.0699 | 2.76 | 3.17 | 5.37 | **0.590** | 0.84 |
| 450 | 1,800 | 0.0550 | 0.0877 | 3.43 | 3.74 | 5.40 | **0.693** | 0.81 |
| 500 | 2,000 | 0.0625 | 0.1051 | 3.94 | 4.29 | 5.45 | **0.787** | 0.80 |
| 550 | 2,200 | 0.0742 | 0.1219 | 4.99 | 5.00 | 5.48 | **0.911** | 0.77 |
| 600 | 2,400 | 0.0993 | 0.1460 | 6.28 | 5.79 | 5.52 | **1.138** | 0.75 |

읽을 때 주의할 점:

- **lag 50 step보다 짧은 간격은 재지 않았다.** 그래서 "뱅크 0"은 "50 step 묵은 feature는 기준을 넘는다"는 뜻이고, 그보다 짧은 간격이 괜찮은지는 이 결과로 알 수 없다. 50 step의 오차 비율 0.141은 기준 0.1과 크게 차이 나지 않는다.
- 학습률을 최대값으로 고정한 측정이라 feature가 가장 빨리 변하는 경우다. 실제 스케줄의 다른 구간에서는 drift가 더 작을 수 있다.
- `뱅크 크기 = lag × 배치 4`다. GradCache-32와 함께 쓰면 update 하나에 32개가 쌓여서, 같은 크기의 뱅크라도 들어 있는 feature의 나이(update 수)는 1/8로 줄어든다. 다만 배치 32에서 update당 drift가 배치 4와 같다는 보장은 없어서 이 표를 그대로 옮겨 쓸 수는 없다.
- lag가 길수록 비교 쌍이 적다(lag 50은 12쌍, lag 600은 1쌍). probe가 32쌍뿐이라 일치 비율 같은 값은 거칠다.
- 0.1이라는 기준은 문헌에서 온 값이 아니라 임의로 정한 휴리스틱이다.

## 파일

- [steptime 보고서](steptime_20261115_132125/steptime_report.json) · [step별 시간](steptime_20261115_132125/step_times.csv) · [영상별 디코딩 시간](steptime_20261115_132125/decode_times.csv)
- [gradcheck 보고서](gradcheck_20261115_132236/gradcheck_report.json)
- [drift 보고서](drift_20261115_132313/drift_report.json) · [lag별 집계](drift_20261115_132313/drift_by_lag.csv) · [시점 쌍별 값](drift_20261115_132313/drift_pairs.csv)
- [콘솔 로그와 종료 코드](logs/) · [실행 환경과 시간](run_summary.json) · [원본 대비 해시](artifact_manifest.json)

로컬 절대 경로는 `<repo>`, `<xpool>`, `<user-home>`으로 바꿨고 숫자는 그대로 두었다. 실행 직전의 `nvidia-smi` 출력은 관련 없는 데스크톱 프로세스 목록이 들어 있어서 내보내지 않았다. GPU 정보만 `run_summary.json`에 옮겼다.
