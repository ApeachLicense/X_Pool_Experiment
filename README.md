# X-Pool Experiment on RTX 3060 Ti

RTX 3060 Ti 8GiB에서 X-Pool의 공식 평가 재현, 시간 제한 학습과 재개, GradCache 배치 32 학습을 검증한 실험 저장소입니다.

**8GiB 환경의 배치 32 학습은 GradCache chunk 4를 기준으로 사용합니다.** MSR-VTT 9K의 두 seed에서 각각 5 epochs를 완료했고, 업데이트 수와 학습률을 맞춘 누적-32 대비 1K-A R@1이 평균 **2.2%p** 높았습니다. [GradCache 비교 결과](https://github.com/ApeachLicense/X_Pool_Experiment/blob/second-experiment/from_now_on_report.md)

**공식 체크포인트의 Recall@1 46.9%를 재현했고, 배치 4로 총 225,000 step의 학습 스케줄을 완료했습니다. 직접 학습한 최종 모델의 Recall@1은 39.9%입니다.** 두 결과는 학습 조건이 다르므로 구분해서 사용해야 합니다.

후속 비교에서는 공식 모델, 기존 epoch 1·2·3·4·5, 초기 6시간 snapshot을 모두 평가했습니다. 이 1K-A 테스트셋에서 자체 모델의 가장 높은 R@1은 **epoch 4의 40.3%**였습니다. 공식 모델만 맞힌 질의 117개 중 100개는 epoch 4에서 정답이 2–5위로 밀렸습니다. [체크포인트 비교](results/checkpoint_analysis_20261109/metrics_summary.csv)와 [실패·회복 사례 7건](results/checkpoint_analysis_20261109/casebook.md)에 결과를 정리했습니다. 이 비교는 독립 검증셋에서 모델을 선택한 결과가 아닙니다.

별도의 **학습 8K·내부 검증 1K**에서 positive cosine margin 0.02를 시험했습니다. 두 seed, 조건당 3,000 updates의 고정 예산에서 원래 loss의 평균 R@1은 **40.40%**, margin은 **39.65%**로 차이는 **−0.75%p**였습니다. 이 설정은 사전에 정한 선별 기준을 통과하지 못해 채택을 보류했습니다. 공식 1K-A와 다른 분할의 결과입니다. [margin 파일럿 결과](results/margin_pilot_20261109/README.md)에 통제 조건·전체 기록·재현 절차를 정리했습니다.

| MSR-VTT 1K-A, text-to-video | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| 공식 9K 체크포인트 평가 | 46.9% | 72.8% | 82.2% |
| 직접 학습한 배치 4, epoch 4 | 40.3% | 68.0% | 77.7% |
| 직접 학습한 배치 4 최종 모델 | 39.9% | 67.7% | 77.7% |

기존 9K 배치 4 학습의 누적 시간은 약 **21시간 55분**, 장치 전체 VRAM 최대 사용량은 **6.59GiB**였으며 메모리 오류는 관측되지 않았습니다. 별도 margin 파일럿의 학습·검증 프로세스 시간 합은 약 **1시간 51분**입니다.

- [상세 실험 보고서와 다음 실험 계획](report.md)
- [체크포인트·질의별 후속 분석](results/checkpoint_analysis_20261109/README.md)
- [Margin 0.02 대조 실험](results/margin_pilot_20261109/README.md)
- [최종 결과 JSON](results/msrvtt9k_split_training/completion_summary.json)
- [최종 체크포인트 검증 기록](results/msrvtt9k_split_training/checkpoint_verification.json)
- [분할 실행 설명](RESUME_TRAINING.md)
- [전체 공개 기록 목록과 SHA-256](results/artifact_manifest.json)

## 코드 출처

기반 코드는 [layer6ai-labs/xpool](https://github.com/layer6ai-labs/xpool)의 커밋 [`bc88718199007d395136adc99a3e69adbee874a0`](https://github.com/layer6ai-labs/xpool/tree/bc88718199007d395136adc99a3e69adbee874a0)입니다. 원 논문은 **X-Pool: Cross-Modal Language-Video Attention for Text-Video Retrieval, CVPR 2022**입니다. 저자, 논문 인용 형식과 원본 사용법은 [원본 README](README_UPSTREAM.md)에 보존했습니다. 이 저장소는 해당 구현을 바탕으로 수행한 별도 실험 기록입니다.

추가 코드는 시간 제한, 측정, 부분 epoch 재개, 셔플·난수 상태 저장과 회귀 테스트를 제공합니다. 모델 구조는 유지했고, 별도 entry point에서 margin loss 대조 실험을 수행했습니다. 이 파일럿에서는 R@1 개선을 확인하지 못했습니다.

## 준비

측정 환경은 Windows, Python 3.9.18, PyTorch 1.8.1+cu111입니다. 다른 라이브러리는 [requirements-local.txt](requirements-local.txt)에 정리했습니다.

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

MSR-VTT 영상은 `data/MSRVTT/vids/`에 별도로 준비해야 합니다. 사용한 압축 영상의 출처는 [Oxford MSRVTT.zip](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip)이며, 원본 저장소의 9K 학습 split과 1K-A 평가 split을 사용했습니다. CLIP 모델·토크나이저는 `openai/clip-vit-base-patch32`를 사용하며, 최초 실행 전에 Hugging Face에서 다운로드할 수 있어야 합니다. 당시 모델 revision과 해시는 [CLIP manifest](results/dataset_manifests/clip_manifest.json)에 있습니다.

공식 평가에는 [공식 9K 체크포인트](https://drive.google.com/file/d/1M2Y41B3a3AxzSJn-n-Xh0Edds97NU1ND/view)를 `outputs/msrvtt9k/model_best.pth`에 배치한 뒤 아래 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe evaluate_local.py
```

## 8GiB에서 GradCache-32 학습

PowerShell에서 새 실험 이름을 지정해 시작합니다.

```powershell
.\.venv\Scripts\python.exe run_training_session.py --hours 7 --exp-name msrvtt9k_gradcache32_s24 --batch-size 32 --grad-cache-chunk-size 4 --seed 24
```

같은 실험 이름과 설정으로 다시 실행하면 `checkpoint-last.pth`에서 이어갑니다. 실행 전에 `--dry-run`으로 데이터 경로와 체크포인트를 확인할 수 있습니다. 실행기는 학습 중 1K-A 평가를 끄며, 최종 체크포인트는 평가 배치 4로 평가합니다. [학습 재개와 평가 명령](RESUME_TRAINING.md)

## Third experiment: SupCon

[Supervised Contrastive Learning (NeurIPS 2020)](https://arxiv.org/abs/2004.11362)의 Eq. (2)를 양방향 text-video contrastive loss로 변형했다. 동일 `video_id`의 모든 교차 모달 조합을 positive로 사용하고, 각 anchor의 positive log-probability를 평균한다. 중복 ID가 없으면 기존 CLIP loss를 그대로 호출한다. 다른 영상 ID의 의미상 일치 여부는 아직 다루지 않는다.

기존 sampler, encoder, pooling, GradCache 및 평가 유사도는 유지한다. 기본 loss는 `clip`으로 남겨 동일 조건 대조 실험을 지원한다. 학습 재개 시 같은 loss 및 실험 이름을 지정한다. 기존 CLIP checkpoint를 SupCon 학습 재개에 사용하지 않는다.

### 1. CPU 회귀 테스트

저장소 루트에서 PowerShell로 실행한다. GPU 학습 전에 아래 테스트가 모두 통과해야 한다.

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_second_experiment.py -v
~~~

검증 항목:

- 모든 video ID가 서로 다르면 기존 CLIP loss 및 gradient와 정확히 일치하는지 확인
- 중복 video ID에 대한 양방향 multi-positive loss 및 gradient가 수식과 일치하는지 확인
- 서로 다른 microbatch에 positive가 있을 때 full-batch와 GradCache의 gradient가 일치하는지 확인
- gradient accumulation 및 학습 실행 명령의 SupCon 옵션 전달 확인

### 2. GPU smoke test (RTX 3060 Ti, 10 updates)

CPU 테스트 통과 후 실제 MSR-VTT 데이터와 CLIP을 사용해 유효 배치 32, microbatch 4로 짧게 실행한다. Smoke test의 출력은 본 학습과 분리한다.

~~~powershell
.\.venv\Scripts\python.exe run_training_session.py --hours 0 --max-steps 10 --exp-name supcon32_smoke_s24 --batch-size 32 --grad-cache-chunk-size 4 --loss supcon --seed 24 --check-recompute
~~~

필요하면 명령 끝에 `--dry-run`을 붙여 데이터 경로와 실행 설정을 먼저 확인한다. `--hours 0`은 시간 제한을 끄고 `--max-steps 10`은 optimizer update를 최대 10회로 제한한다.

통과 기준: 정상 종료, 10 updates 완료, 유한한 loss, GradCache feature 재계산 검사 통과, CUDA OOM 없음, checkpoint 저장 확인. `outputs/supcon32_smoke_s24/`의 training log, monitor report, training profile 및 checkpoint를 확인한다. 무작위 배치에서 중복 ID가 없을 수도 있으므로 실제 다중 positive 처리의 정확성은 CPU 테스트로 별도 검증한다. Smoke test는 성능 평가가 아니며 본 학습에 해당 체크포인트를 사용하지 않는다.

### 3. 본 학습 (Seed 24, 25)

기존 GradCache-32 비교 실험과 동일하게 두 seed에서 각각 5 epochs를 학습한다.

| 항목 | 설정 |
| --- | --- |
| 데이터 | MSR-VTT 9K train / 1K-A 최종 평가 |
| 유효 배치 / microbatch | 32 / 4 |
| Seed | 24, 25 |
| Epochs | 5 |
| Optimizer updates | seed당 28,125 |
| CLIP / pooling learning rate | 1e-6 / 3e-5 |
| Loss | SupCon |
| 초기화 | 기존 사전학습 CLIP 및 새 X-Pool pooling |
| 세션 | 7시간 단위, 동일 설정으로 재개 |

Seed 24:

~~~powershell
.\.venv\Scripts\python.exe run_training_session.py --hours 7 --exp-name supcon32_s24 --batch-size 32 --grad-cache-chunk-size 4 --loss supcon --seed 24
~~~

Seed 25:

~~~powershell
.\.venv\Scripts\python.exe run_training_session.py --hours 7 --exp-name supcon32_s25 --batch-size 32 --grad-cache-chunk-size 4 --loss supcon --seed 25
~~~

8GiB GPU 한 장에서 두 실행을 동시에 시작하지 않는다. 각 seed의 첫 실행은 새 실험 디렉터리에서 시작하고, 시간이 끝나면 같은 명령을 다시 실행해 해당 실험의 `checkpoint-last.pth`에서 5 epochs 완료까지 이어간다. Smoke test나 기존 CLIP loss의 체크포인트로 재개하지 않는다. 완료 여부는 training profile 및 checkpoint의 epoch/update 기록으로 확인한다.

### 4. 비교 및 최종 평가

기존 GradCache-32 CLIP loss의 seed별 결과를 기준으로 비교한다. 같은 seed에서 초기화, 데이터 순서, optimizer, learning rate schedule, epoch 및 update 예산을 맞춘다. 기존 기록의 조건이 일치하지 않거나 가중치가 없다면 별도 baseline 실행을 고려한다. 새 대조군이 필요한 경우 위 명령에서 `--loss clip`으로 바꾸고 각각 `clip32_control_s24`, `clip32_control_s25`처럼 별도 실험 이름을 사용한다.

| Seed | 기존 GradCache-32 R@1 | 기존 R@5 | 기존 R@10 | SupCon R@1 | SupCon R@5 | SupCon R@10 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 24 | 45.4% | 72.5% | 81.6% | 미측정 | 미측정 | 미측정 |
| 25 | 46.1% | 72.9% | 82.0% | 미측정 | 미측정 | 미측정 |

학습 중 1K-A 평가를 사용하지 않고 각 seed의 최종 5번째 epoch 체크포인트만 같은 평가 프로토콜(평가 배치 4)로 평가한다. [학습 재개와 평가 명령](RESUME_TRAINING.md)을 참고한다. R@1/R@5/R@10의 seed별 차이 및 평균 차이, VRAM peak, 학습 시간, training profile의 duplicate_video_batches를 기록한다.

동일 video ID가 없는 배치에서는 기존 CLIP loss와 동일하므로 중복 빈도가 낮으면 개선 폭도 제한적일 수 있다. sampler 변경이나 하이퍼파라미터 선택은 별도 내부 검증셋을 이용한 ablation으로 분리하고 1K-A 결과에 맞춰 조정하지 않는다.

아직 새 손실의 CPU/GPU 테스트 결과 및 학습 성능은 측정하지 않았다. 후속 논문 및 실험 계획은 [TODO.md](TODO.md)를 참고한다.

## 공개 파일 범위

코드, 기존 split·annotation, 실행 설정, 전체 step CSV, GPU 측정 CSV, 로그, 요약 JSON, 초기 6시간 VRAM 그래프를 포함합니다. 영상, 가상환경, 다운로드한 CLIP 가중치와 학습 체크포인트는 포함하지 않습니다. 최종 체크포인트의 해시는 검증 기록에 남아 있으며, 가중치 파일은 원래 실험 PC에 보관되어 있습니다.

`results/`는 과거 실행 기록의 공개 사본입니다. JSON 내부의 `outputs/...`는 당시 실행 경로를 뜻하며 공개 사본은 `results/...`에 있습니다. 개인 PC의 절대 경로는 `<repo>` 또는 `<user-home>`으로 치환했고, 수치와 CSV는 유지했습니다. 원본·공개 사본의 해시는 artifact manifest에서 확인할 수 있습니다.

## 재개 코드 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

기존 실제 GPU 재개 비교 결과는 [resume_gpu_verification.json](results/resume_gpu_verification.json)에 있습니다. 공개용 복사본에서 다시 실행한 CPU 회귀 테스트 기록은 `results/validation/`에 보관합니다.
