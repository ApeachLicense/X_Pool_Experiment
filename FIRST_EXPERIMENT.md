# 긴 학습 전 사전 확인 3종 (`first_experiment`)

GradCache·memory bank 실험에 22시간짜리 학습을 쓰기 전에, 몇 분짜리 확인 세 가지로 전제를 먼저 검증한다.

| 명령 | 답하려는 질문 | 예상 시간(추정) |
| --- | --- | --- |
| `gradcheck` | 배치 4 전체의 gradient와 2+2 chunk GradCache의 gradient가 일치하는가 | 1~2분 |
| `steptime` | step 시간 중 영상 디코딩과 GPU 연산의 비중은 얼마인가 | 1~2분 |
| `drift` | 저장해 둔 CLIP feature가 몇 step 뒤까지 쓸 만한가 (뱅크 크기) | 5분 안팎 |

예상 시간은 기존 측정값(배치 4, step당 약 0.35초)으로 계산한 추정이고 모델 로딩 시간은 뺐다.

세 명령 모두 체크포인트를 저장하지 않고 기존 실험 폴더를 건드리지 않는다. 결과는 `outputs/first_experiment/<명령>_<UTC 시각>/`에 생긴다.

## 준비

기존 학습과 같은 환경(`requirements-local.txt`), `data/MSRVTT/vids/`의 영상, `openai/clip-vit-base-patch32` 가중치가 필요하다. 데이터가 이 저장소 밖에 있으면 `--source-root <폴더>`로 지정한다. 옵션은 명령 이름 뒤에 쓴다.

## 1. gradcheck

```powershell
.\.venv\Scripts\python.exe first_experiment.py gradcheck
```

학습 로더에서 배치 4를 꺼내 같은 배치로 세 가지 gradient를 계산하고 비교한다.

- **full**: 평소처럼 배치 4를 한 번에 forward/backward한 기준값.
- **grad_cache**: 2개씩 no-grad 인코딩 → 4×4 pooling과 loss 한 번 → 2개씩 다시 인코딩하며 저장한 gradient 전달.
- **accumulation**: chunk마다 따로 loss를 계산하는 단순 gradient accumulation. 대조용이다.

읽는 법:

- `grad_cache rel_err`가 허용치(기본 `1e-2`) 이하면 `GRADCHECK PASSED`다. 실제 크기의 무작위 가중치 모델을 CPU FP32로 돌렸을 때는 `1e-7` 수준이었다.
- `accumulation rel_err`는 1에 가깝게 나오는 것이 정상이다. chunk 안의 negative만 보기 때문이고, 이 값이 작게 나오면 오히려 테스트를 의심해야 한다.
- `feature_recompute_max_abs_diff`는 같은 샘플을 두 번 인코딩한 결과의 차이다. 부동소수 오차 수준을 넘으면 인코더에 난수나 배치 의존성이 있다는 뜻이고, 그 경우 GradCache는 정확하지 않다.
- `forward == pool(encode)`는 `CLIPTransformer.forward`를 `encode`/`pool`로 나눈 뒤에도 출력이 비트 단위로 같은지 본다.
- loss가 `--min-loss`(기본 `1e-4`)보다 작은 배치는 gradient가 거의 0이라 판정에서 뺀다. 판정에 쓸 배치가 `--num-batches`개(기본 3) 모일 때까지 뽑는다.
- 각 방식의 시간과 peak VRAM도 `gradcheck_report.json`에 남는다.

학습된 가중치에서도 확인하려면 `--checkpoint <경로>`를 붙인다.

## 2. steptime

```powershell
.\.venv\Scripts\python.exe first_experiment.py steptime
```

실제 학습과 같은 로더·모델·optimizer로 100 step(앞 10 step 제외)을 돌리며 한 step을 네 구간으로 잰다.

| 구간 | 의미 |
| --- | --- |
| `data_wait` | 메인 프로세스가 로더를 기다린 시간. worker가 GPU를 따라오면 0에 가깝다 |
| `tokenize` | 캡션 토크나이즈 |
| `to_device` | 텐서를 GPU로 복사 |
| `gpu_step` | forward, backward, optimizer update |

이어서 영상 40개를 프로세스 하나로 직접 읽어 영상 1개당 디코딩·전처리 시간을 잰다.

읽는 법:

- `VERDICT`가 `gpu-bound`면 프레임을 미리 추출해도 얻을 것이 거의 없다. `data-bound`면 미리 추출하는 쪽이 모든 실험을 빠르게 한다.
- 판정 기준은 두 가지다. 로더가 초당 만들 수 있는 샘플 수(`worker 수 ÷ 영상 1개 처리 시간`)가 GPU가 소비할 수 있는 수보다 적거나, `data_wait`가 step의 20% 이상이면 `data-bound`다.
- `speedup_if_data_wait_removed`는 로더 대기가 사라졌을 때의 최대 속도 향상이다.
- 마지막 줄의 GradCache 예상 시간은 "기존 step + chunk당 no-grad 인코딩 1회"로 계산한 **추정**이다. 배치 32짜리 로더의 디코딩 부하는 반영하지 않았다.

worker 수를 바꿔 보려면 `--num-workers 4`처럼 준다.

## 3. drift

```powershell
# (a) 사전학습 CLIP에서 시작, 최대 학습률 고정: 가장 빨리 변하는 경우
.\.venv\Scripts\python.exe first_experiment.py drift

# (b) 학습 중간 체크포인트에서 시작: 그 시점의 optimizer 상태와 학습률 사용
.\.venv\Scripts\python.exe first_experiment.py drift --checkpoint outputs\<실험 폴더>\checkpoint-epoch1.pth
```

학습 영상 32개를 probe로 고정하고(중앙 crop, 영상마다 첫 캡션), 배치 4로 평소처럼 600 step 학습하면서 50 step마다 probe의 feature를 다시 뽑아 이전 시점들과 비교한다.

읽는 법 (`drift_by_lag.csv`, 콘솔 표):

- `lag_steps`는 두 시점의 간격이고 `bank_entries = lag_steps × 배치 크기`는 가장 오래된 항목의 나이가 그만큼인 뱅크 크기다.
- `frame_cos_dist`, `text_cos_dist`는 feature 자체의 변화다.
- `logit_err_video`는 현재 텍스트와 **옛 프레임 feature**로 계산한 logit이 최신 값과 얼마나 다른지다. text→video loss가 뱅크에서 받는 오차다. `logit_err_text`는 반대 방향이다. pooling은 뱅크 설계와 같이 항상 현재 모듈로 다시 계산한다.
- `neg_logit_std`는 negative logit의 퍼짐이다. 오차를 이 값과 비교해야 크기를 판단할 수 있다.
- `hardest_neg_agree`는 옛 feature로도 가장 어려운 negative가 같은 영상으로 뽑히는 비율이다.
- `SUGGESTED max bank entries`는 "평균 logit 오차 ≤ `--tolerance-ratio`(기본 0.1) × `neg_logit_std`"를 만족하는 가장 큰 뱅크 크기다. **이 규칙은 문헌 값이 아니라 임의로 정한 휴리스틱**이므로 표의 숫자를 직접 보고 판단한다.

주의:

- 학습률이 결과를 좌우한다. 실제 스케줄은 warmup이 22,500 step이라 처음 600 step의 학습률은 거의 0이다. 그래서 기본값은 스케줄 대신 최대 학습률을 고정한다(`--lr-mode peak`). 실제 스케줄 그대로 보려면 `--lr-mode schedule`을 준다.
- 마지막 epoch 체크포인트는 cosine 스케줄 끝이라 학습률이 0이다. 이 경우 경고가 뜨고 feature가 변하지 않으니 더 이른 체크포인트를 쓴다.
- probe는 32쌍이라 `hardest_neg_agree` 같은 비율은 거칠다.

## 코드 구성

| 파일 | 내용 |
| --- | --- |
| `model/clip_transformer.py` | `encode()`(CLIP 인코딩)와 `pool()`(text-conditioned pooling) 추가. `forward()`는 둘을 차례로 부르며 연산과 순서는 그대로다 |
| `modules/grad_cache.py` | `grad_cache_backward()`: chunk 단위로 큰 배치의 gradient를 누적. 이후 학습 루프에서 그대로 쓸 수 있다 |
| `modules/feature_drift.py` | snapshot 비교, lag별 집계, 뱅크 크기 규칙 |
| `first_experiment.py` | 세 명령의 실행 진입점 |
| `tests/test_first_experiment.py` | 단위 테스트 16개 |

`trainer/trainer.py`와 `train.py`는 수정하지 않았다. GradCache를 학습 루프에 넣는 작업은 `gradcheck`가 통과한 뒤의 일이다.

## 테스트와 검증 범위

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Python 3.9.25, PyTorch 1.8.1(CPU 실행), Transformers 4.6.1에서 기존 13개와 새 16개가 모두 통과했다. 새 테스트는 구조가 같은 작은 무작위 CLIP으로 다음을 확인한다.

- `forward()`와 `pool(encode())`의 출력이 비트 단위로 같다.
- float64에서 GradCache gradient가 chunk 크기 1·2·3 모두 전체 배치와 `1e-9` 이내로 일치한다(CLIP 텍스트·비전, pooling, logit scale 각각).
- pooling dropout 마스크가 다르거나, chunk별 loss를 쓰거나, 인코더에 난수가 있으면 비교가 실패한다.
- 세 명령이 실제 로더와 OpenCV 디코딩을 거쳐 끝까지 실행되고 보고서를 쓴다. 테스트는 worker 0개로 돌고, worker 2개로는 따로 한 번 실행해 확인했다.

추가로 ViT-B/32와 같은 크기의 무작위 가중치 모델(1.5억 파라미터, 12프레임, 224px, logit scale 100)을 CPU FP32로 돌렸을 때 GradCache의 상대 오차는 `1.2e-7`, 단순 accumulation은 `0.85~0.90`이었다.

확인하지 못한 것:

- 실제 사전학습 가중치와 MSR-VTT 영상을 쓴 GPU 실행. 이 환경에는 GPU, 영상, 가중치가 없었다.
- Windows에서의 실행. 테스트는 Linux에서만 돌렸다.
- GPU에서의 시간·VRAM 수치. 문서의 예상 시간은 모두 추정이다.
