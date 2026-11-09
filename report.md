# X-Pool 실행·학습 재현 실험 보고서

실험 기간: 2026-11-07–2026-11-09. 본문 시각은 한국 표준시(KST, UTC+9)이며 원시 JSON의 시각은 UTC입니다.

## 확인된 결과

**공식 사전 학습 모델의 MSR-VTT 1K-A Recall@1 46.9%를 재현했다. RTX 3060 Ti 8GiB에서는 배치 4, FP32로 학습 스케줄을 끝까지 실행할 수 있었다. 직접 학습한 마지막 모델의 Recall@1은 39.9%로, 공식 모델보다 7.0%p 낮았다.**

따라서 현재 확보한 것은 공식 모델 평가의 재현 결과와 소형 GPU에서의 학습 가능성이다. 직접 학습한 모델이 원 논문의 학습 성능을 재현했다고 주장할 수는 없다. 다음 단계에서는 공식 체크포인트를 성능 기준으로 유지하고, 성능 차이의 원인과 검색 실패 사례를 조사하는 것이 적절하다.

| 검증 항목 | 결과 | 근거 |
| --- | --- | --- |
| 공식 체크포인트 평가 | R@1 46.9%, 정상 종료 | [평가 JSON](results/msrvtt9k/evaluation_report.json) |
| 8GiB GPU 학습 가능성 | 배치 4, FP32, 메모리 오류 미관측 | [누적 결과](results/msrvtt9k_split_training/completion_summary.json) |
| 전체 학습 스케줄 | 5 epochs에 해당하는 225,000 optimizer updates | [체크포인트 검증](results/msrvtt9k_split_training/checkpoint_verification.json) |
| 분할 실행 재개 | 모델·optimizer·scheduler·셔플·난수 상태 저장 | [GPU 비교](results/resume_gpu_verification.json) |
| 최종 모델 평가 | R@1 39.9%, 정상 종료 | [최종 평가 JSON](results/msrvtt9k_split_training/final_evaluation.json) |
| Cap4Video | 실행하지 않음 | 본 보고서는 X-Pool 실험만 다룸 |

## 환경과 데이터

| 항목 | 사용 조건 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Ti, 8192MiB |
| 시스템 메모리 / 운영체제 | 32GB / Windows, PowerShell |
| Python | 3.9.18 |
| PyTorch / torchvision | 1.8.1+cu111 / 0.9.1+cu111 |
| Transformers / OpenCV | 4.6.1 / 4.5.3.56 |
| NumPy / pandas / SciPy | 1.22.4 / 1.3.5 / 1.7.3 |
| 기반 코드 | [공식 저장소](https://github.com/layer6ai-labs/xpool), 커밋 `bc88718199007d395136adc99a3e69adbee874a0` |
| CLIP | `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| 학습 데이터 | MSR-VTT 9K, 9,000 videos / 180,000 caption-video pairs |
| 평가 데이터 | MSR-VTT 1K-A, 1,000 text-video pairs, text-to-video retrieval |
| 입력 | 영상당 12 frames, uniform sampling, 224×224 |

9K 학습 split과 1K-A 평가 split의 영상 ID는 서로 겹치지 않는다. 총 10,000개 영상 파일의 다운로드와 CRC32 확인 기록을 보관했다. [학습 manifest](results/dataset_manifests/msrvtt_9k_train_manifest.json), [평가 manifest](results/dataset_manifests/msrvtt_1ka_manifest.json), [CLIP manifest](results/dataset_manifests/clip_manifest.json)를 참고한다.

## 공식 모델 평가

원본 README에서 제공한 MSR-VTT 9K 체크포인트를 사용했다. 평가 배치만 4로 낮췄으며, 원본 README는 평가 배치 크기를 변경해도 결과가 달라지지 않는다고 설명한다. 학습 배치 크기를 바꾸는 경우와는 구분해야 한다.

| 항목 | 측정값 |
| --- | ---: |
| R@1 / R@5 / R@10 | 46.9% / 72.8% / 82.2% |
| MedR / MeanR | 2.0 / 14.291 |
| 실행 시간 | 138.704초, 약 2분 19초 |
| 장치 VRAM 최대 사용량 | 4461MiB, 약 4.36GiB |
| 종료 코드 | 0 |

이 결과는 제공된 공식 모델과 평가 데이터 준비가 정상임을 뒷받침한다. 직접 학습을 재현했다는 근거로 사용하지는 않는다. [평가 로그](results/msrvtt9k/evaluation.log)에 원래 출력이 있다.

## 직접 학습 설정

| 항목 | 값 |
| --- | --- |
| 학습 배치 / 정밀도 | 4 / FP32 |
| 학습 스케줄 | 5 epochs, epoch당 45,000 updates, 총 225,000 updates |
| CLIP learning rate | 1e-6 |
| 새 pooling 모듈 learning rate | 3e-5 |
| Weight decay / warmup proportion | 0.2 / 0.1 |
| Transformer dropout / heads | 0.3 / 1 |
| Seed / data workers | 24 / 2 |
| 학습 중 평가 | 비활성화 (`--evals_per_epoch=0`) |
| 메모리 절감 추가 기법 | AMP, gradient checkpointing, gradient accumulation 적용하지 않음 |

공식 README의 MSR-VTT 9K 학습 배치는 32다. 이 실험은 배치 4를 사용했으므로 contrastive loss의 배치 내 negative 구성이 다르다. 배치 차이, 검증에 따른 모델 선택 차이, 중단·재개 조건 등을 고려해야 하며, 한 번의 실행만으로 성능 차이의 원인을 특정할 수 없다.

## 실행 이력과 시간

| 실행 | 학습 시작(KST) | 학습·저장 시간 | 추가 updates | 누적 updates | 장치 VRAM 최대 |
| --- | --- | ---: | ---: | ---: | ---: |
| 초기 6시간 | 11/07 14:48경 | 21,603.938초 | 62,216 | 62,216 | 6751MiB / 6.59GiB |
| 두 번째 7시간 | 11/08 13:17경 | 25,207.640초 | 72,465 | 134,681 | 6748MiB / 6.59GiB |
| 남은 학습, 최대 10시간 예산 | 11/09 06:49경 | 32,122.453초 | 90,319 | 225,000 | 6646MiB / 6.49GiB |
| 합계 | — | **78,934.031초, 약 21시간 55분** | **225,000** | **225,000** | **6.59GiB** |

마지막 학습은 11월 9일 15:45경 정상 종료했다. 10시간 예산을 모두 사용하기 전에 학습 목표에 도달했다. 이후 최종 평가는 147.192초가 걸렸고 15:47경 완료됐다.

위 시간은 학습 프로파일 시작부터 마지막 체크포인트 저장까지다. 데이터 다운로드, 환경 구성, 세션 사이 대기 시간, 별도 평가, 짧은 준비 검증은 합계에 포함하지 않는다. 프로세스 전체 실행 시간은 각 `monitor_report.json`의 `wall_time_seconds`에서 별도로 확인할 수 있다.

전체 학습에서 900,000개의 caption-video pair를 처리했다. 이는 반복 처리 횟수이며 고유 샘플 수가 아니다. 전 구간 step CSV를 확인한 결과 step 1–225,000이 연속이고 모든 loss가 유한했다. 메모리 오류는 관측되지 않았다. 초기 짧은 배치 탐색 기록도 `results/vram_probe_b2_20261107_054714/`와 `results/vram_probe_b4_20261107_054714/`에 포함했다.

## VRAM 측정의 범위

`nvidia-smi`로 약 5초 간격의 장치 전체 사용량을 기록했다. **6.59GiB는 데스크톱과 다른 프로세스 사용량을 포함한 관측 최대값**이며 모델 tensor 메모리만의 값이 아니다. 샘플 사이의 순간 최대 사용량까지 보장하는 측정은 아니다.

초기 6시간 실행에서 PyTorch의 peak allocated는 약 4.34GiB, peak reserved는 약 4.58GiB였다. 전체 세션 중 장치 온도 최대값은 73.31°C였다. 이 수치는 위 설정에서의 결과이며, 배치 32나 다른 영상 길이에서의 실행 가능성은 검증하지 않았다.

![초기 6시간 VRAM 측정](results/msrvtt9k_train_vram_20261107_054714/vram_6h.png)

## 분할 실행과 재개 검증

`ResumableBatchSampler`는 epoch의 실제 셔플 순열과 다음 완료할 배치 위치를 저장한다. worker prefetch는 완료 위치를 전진시키지 않고, optimizer update가 끝난 뒤에만 위치를 갱신한다. 샘플별 증강은 epoch와 index에 따른 seed를 사용한다.

체크포인트에는 모델, optimizer, scheduler, global step, Python·NumPy·PyTorch CPU·CUDA 난수 상태를 함께 저장한다. 임시 파일을 저장한 후 교체하고, 동일 실험 폴더에 두 학습 프로세스가 동시에 쓰지 못하도록 잠금을 둔다. 각 세션의 시간 제한은 재개할 때 새로 계산하며 기존 로그와 CSV는 보존한다.

**초기 6시간의 legacy 체크포인트에는 당시 셔플과 난수 상태가 없었다.** 따라서 62,216 step에서 처음 재개할 때부터 새로운 결정적 셔플 규칙을 적용했다. 이 전환을 가로지르는 학습이 처음부터 한 번에 연속 실행한 학습과 동일하다고 보장할 수 없다. 특히 중단된 epoch 2의 샘플 순서에 차이가 있을 수 있다. 이후 재개부터는 실제 셔플·난수 상태를 저장했다.

검증 범위는 다음과 같다.

- CPU 회귀 테스트 5개: 여러 번 재개, worker 수 변경, epoch 경계·마지막 짧은 배치, 세션 시간 초기화·로그 보존, legacy 전환, 설정 불일치와 동시 쓰기 거부를 검사했다. 공개용 클론에서도 5개 모두 통과했다([검증 기록](results/validation/cpu_resume_tests.json)). 새 클론에서 테스트 임시 폴더의 부모인 `outputs/`를 생성하도록 테스트 준비 코드를 보완했으며, 공개 정리 과정에서 모델·학습 코드는 변경하지 않았다.
- 실제 X-Pool GPU: 4 updates 연속 실행과 2 updates 후 재개한 2 updates를 비교했다. 416개 모델 tensor, optimizer, scheduler, 난수 및 셔플 상태가 비트 단위로 일치했다. 이는 짧은 비교 실험의 결과이며 전체 22시간 연속 실행과 비교한 것은 아니다.
- 최종 체크포인트: epoch 5, global step 225,000, `partial_epoch=false`, 다음 배치 45,000, optimizer와 scheduler의 step 일치, 모델 tensor의 유한값, 난수 상태 보존을 확인했다. `checkpoint-last.pth`와 epoch 5 파일의 SHA-256도 일치했다.

## 최종 검색 성능

동일한 1K-A text-to-video 평가 프로토콜에서 비교했다. Recall은 높을수록 좋고 MedR·MeanR은 낮을수록 좋다.

| 모델 | R@1 | R@5 | R@10 | MedR | MeanR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 공식 9K 체크포인트 | 46.9% | 72.8% | 82.2% | 2.0 | 14.291 |
| 직접 학습한 배치 4, epoch 5 | 39.9% | 67.7% | 77.7% | 2.0 | 14.002 |

최종 모델은 R@1이 7.0%p 낮다. MeanR은 소폭 낮지만, 이것만으로 전체 검색 성능이 개선됐다고 볼 수 없다. 학습 중 평가를 끈 상태로 마지막 epoch를 평가했으므로 검증 성능으로 선택한 최적 모델의 결과도 아니다. [최종 평가 로그](results/msrvtt9k_split_training/final_evaluation.log)에 출력이 있다.

최종 파일은 원래 실험 PC의 `outputs/msrvtt9k_split_training/checkpoint-epoch5.pth`에 보관했다. 크기는 1,833,014,183 bytes이며 SHA-256은 다음과 같다. 이 파일은 Git 저장소에는 포함하지 않았다.

```text
cae2e62e18282026e2e6d9703cc8d0ef5e77f9b7dcb795685a088aec86ddfd8a
```

## 다음 실험의 권장 순서

1. **공식 체크포인트를 성능 기준으로 고정한다.** 이미 46.9% 평가를 재현했으므로 이 모델의 query별 정답 rank, 상위 검색 영상, 실패 사례와 attention을 추출하는 작업부터 진행한다. 직접 학습한 39.9% 모델은 별도의 소형 배치 실험으로 관리한다.
2. **성능 차이의 원인 후보를 짧은 비교 실험으로 좁힌다.** seed, 데이터 split, 입력, 총 학습 예산과 검증 규칙을 명시하고 한 조건씩 바꾼다. 우선 배치 4의 중간 체크포인트를 같은 방식으로 평가해 마지막 epoch 선택의 영향을 확인할 수 있다. 그다음 메모리가 허용하는 배치와 정밀도 설정을 짧게 측정하고, 후보를 고른 뒤 전체 학습 여부를 결정한다. 이 비교 실험은 아직 실행하지 않았다.
3. **실패 유형에 근거한 개선안을 선택한다.** 예를 들어 관련 프레임에 attention이 분산되는 사례가 반복되면 pooling이나 frame 선택을 후보로 삼을 수 있다. 이는 분석 후 검증할 가설이며 현재 성능 개선의 증거는 없다. Cap4Video나 새 caption generator 도입은 그 이후 별도의 baseline 검증으로 다룬다.

Gradient accumulation은 단순히 optimizer update 간격을 늘리는 구현만으로는 각 forward의 contrastive negative 수를 늘려 주지 않는다. 따라서 배치 4를 8번 누적했다는 이유만으로 논문의 배치 32와 동등하다고 간주하면 안 된다. 더 큰 negative 집합을 만드는 방식은 별도로 구현하고 검증해야 한다.

## 재현 절차

아래 명령은 사용 방법을 기록한 것이다. 보고서를 작성하고 공개하는 과정에서는 추가 GPU 학습이나 평가를 실행하지 않았다. 환경·데이터·공식 가중치 준비는 [README](README.md#준비)를 따른다.

공식 모델 평가:

```powershell
.\.venv\Scripts\python.exe evaluate_local.py
```

새로운 배치 4 학습을 처음부터 최대 7시간 실행하는 예시:

```powershell
.\.venv\Scripts\python.exe train.py `
  --exp_name=msrvtt9k_new_run --videos_dir=data/MSRVTT/vids `
  --dataset_name=MSRVTT --msrvtt_train_file=9k --huggingface `
  --batch_size=4 --num_workers=2 --num_epochs=5 `
  --noclip_lr=3e-5 --transformer_dropout=0.3 `
  --evals_per_epoch=0 --log_step=100 --no_tensorboard `
  --profile_training --max_train_seconds=25200
```

위 새 실험을 저장된 위치부터 최대 10시간 더 실행하는 예시:

```powershell
.\.venv\Scripts\python.exe run_training_session.py --exp-name msrvtt9k_new_run --hours 10
```

실측 전체 시간은 약 22시간이므로 새 학습이 7+10시간만으로 끝난다고 가정하면 안 된다. 기록된 이번 실험은 그 전에 수행한 6시간을 포함해 완료했다. 최대 5 epoch까지만 학습하고, 시간 제한이 먼저 도달하면 다음 세션으로 이어간다.

완료된 로컬 모델 평가 예시(해당 가중치를 별도로 준비해야 함):

```powershell
.\.venv\Scripts\python.exe test.py `
  --exp_name=msrvtt9k_split_training --videos_dir=data/MSRVTT/vids `
  --dataset_name=MSRVTT --msrvtt_train_file=9k --huggingface `
  --batch_size=4 --num_workers=2 --load_epoch=5 --no_tensorboard
```

재개 회귀 테스트:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 공개 기록의 구성과 한계

- `results/msrvtt9k/`: 공식 모델 평가 결과와 로그.
- `results/msrvtt9k_train_vram_20261107_054714/`: 초기 6시간 실행의 요약, step·GPU CSV, 로그와 그래프.
- `results/msrvtt9k_split_training/`: 7시간·최종 실행의 기록, 최종 평가, 체크포인트 검증과 전체 요약. 루트의 `session_summary.json`과 step CSV는 7시간 실행 기록이며 최신 프로파일은 최종 실행 기록이다. 추가 세션의 상세 기록은 `sessions/`에 있다.
- `results/resume_gpu_verification.json`, `results/resume_gpu_*`, `results/resume_session_entry_test/`: 재개 구현 검증 기록.
- `results/dataset_manifests/`: 영상·CLIP 다운로드 확인 자료. 영상이나 모델 가중치 자체는 포함하지 않는다.
- `results/validation/`: 공개용 복사본에서 수행한 CPU 회귀 테스트 기록.
- `results/artifact_manifest.json`: 원본·공개 사본의 해시와 경로 치환 여부.

공개 사본의 로그·JSON에서는 개인 PC의 절대 경로를 치환했다. 수치와 원시 CSV는 유지했다. 모델 가중치, 영상, 가상환경은 로컬에만 보관했다. 일부 보조 패키지는 `requirements-local.txt`에서 버전을 고정하지 않았으며, 다른 GPU·드라이버·라이브러리에서 비트 단위 동일성을 보장하지 않는다. 여러 seed의 평균이나 통계적 유의성 실험도 수행하지 않았다.
