# Second experiment — GradCache-32와 누적-32

## 연구 질문

RTX 3060 Ti에서 더 큰 대조 배치를 쓰면 MSR-VTT 1K-A 검색 순위가 개선되는가? 자체 배치 4 모델의 마지막 epoch는 R@1 **39.9%**, 공식 가중치는 **46.9%**였다. 이 7.0%p 차이가 실험의 동기다. 우선 배치 크기 변경의 효과를 보고, 이어서 같은 업데이트 스케줄에서 더 넓은 negative 집합을 쓰는 효과를 확인한다.

공식 **46.9%는 참고값**이다. 원본 코드 기본 설정은 1K-A를 epoch마다 10회 평가해 R@1의 이동 평균이 가장 좋은 체크포인트를 `model_best.pth`에 저장한다. 공개 파일명도 `model_best.pth`다. 정확한 선택 이력은 확인되지 않았고, 1K-A를 반복해서 모델 선택에 사용했다면 최종 epoch를 한 번 평가한 값보다 낙관적일 수 있다. 비교에 쓸 모든 실행은 학습 중 1K-A 평가를 끈 뒤 **고정된 epoch 5 체크포인트만 1K-A에서 한 번 평가**한다. [원본 실행법](README_UPSTREAM.md) · [배치 4 최종 기록](results/msrvtt9k_split_training/final_evaluation.json)

## 비교 설계

| 조건 | 업데이트당 노출 | 한 loss의 후보 / query당 negative 슬롯 | 5 epoch 업데이트 |
| --- | ---: | ---: | ---: |
| 배치 4 | 4쌍 | 4 / 3 | 225,000 |
| 누적-32 | 32쌍 | 4 / 3 | 28,125 |
| GradCache-32 | 32쌍 | 32 / 31 | 28,125 |

- **GradCache-32 vs. 배치 4**는 전체 설정의 효과를 본다. Negative 수, 업데이트 수, 샘플당 업데이트 빈도와 학습률 적용량이 함께 달라진다. 기존 39.9%는 같은 최종 평가 정책의 비교점이며, 이전 실행은 중간에 셔플 규칙을 바꿨다. 인과 비교와 seed 반복에는 현 브랜치에서 새 배치 4 대조군을 쓴다.
- **GradCache-32 vs. 누적-32**는 seed별 초기화, 32쌍의 샘플 순서와 증강, 노출량, 업데이트 수, 업데이트별 LR과 optimizer를 맞춘다. GradCache는 텍스트마다 32개 영상을 pooling해 32×32 loss를 한 번 계산한다. 누적-32는 4쌍씩 독립 loss를 8회 계산하고 예제 수로 가중 평균한 gradient로 한 번 업데이트한다. 마지막 청크가 짧으면 실제 크기로 가중한다. 둘 다 clipping, optimizer, scheduler, global step은 32쌍마다 한 번 움직인다. 이 비교는 **배치 전체를 함께 대조하는 목적식의 효과**를 본다. Dropout mask와 같은 영상이 오답으로 포함되는 빈도도 달라질 수 있다. 이전 gradcheck에서 누적 gradient의 상대 오차가 약 0.84–1.04였던 점은 두 목적식이 다르다는 확인이다. [이전 gradcheck](results/first_experiment_20261115/README.md)
- 초기화는 기본 CLIP ViT-B/32와 원래 pooling 초기화, MSR-VTT 9K, FP32 입력·가중치, 12프레임·224px, 기존 증강을 유지한다. 학습률은 CLIP 1e-6 / pooling 3e-5, warmup 10%와 cosine 스케줄을 쓴다. TF32 플래그를 기록하고 조건 간 같은 환경을 쓴다. seed 24로 시작하고, 효과를 판단할 때 seed 25로 짝을 맞춰 반복한다.

**판정 기준.** 주 지표는 고정 epoch 5의 text-to-video R@1 차이다. R@5, R@10, MedR, MeanR도 함께 보고한다. 두 seed의 짝지은 R@1 차이가 같은 양의 방향이고 평균이 **+1.0%p 이상**이면 해당 대비에서 개선 신호로 본다. 단일 seed의 약 1%p 이내 변화, seed 간 방향 불일치, 두 seed 평균 +1.0%p 미만은 이 실험의 기준에서 불확실하다. Margin pilot의 내부 분할에서는 seed 간 표본 표준편차가 약 0.4–0.6%p였다. 이 수치를 1K-A의 오차 추정치로 옮겨 쓰지 않고, 작은 차이에 반복이 필요하다는 참고로 사용한다. 위 기준은 실용적 선별 기준이다. [Margin pilot 기록](results/margin_pilot_20261109/README.md)

XBM은 별도 후보로 남긴다. 배치 4 drift에서 오차 비율 0.1 이내 뱅크는 사전학습 직후 최대 LR에서 40개, epoch 1에서 800개, epoch 3에서 2,400개 이상이었다. Epoch 1 lag 200에서도 가장 어려운 negative 일치는 92%였고 오차 꼬리가 컸다. Warmup 종료 지점은 미측정이며, 배치 4 기준 lag를 배치 32로 환산할 수 없다. 현재 가중치에서 매번 feature를 다시 구하는 GradCache-32를 먼저 검증한다. 이후 배치 4 + XBM을 시험할 때는 실제 warmup 종료 부근에서 drift를 다시 재고 bank 크기·시작 시점을 정한다. LP-FT, SigLIP는 이번 범위 밖이다. [Drift 재검증](results/first_experiment_20261115/README.md)

## Windows에서 smoke 실행

`X_Pool_Experiment`와 `xpool`이 나란히 있는 환경에서 `run_second_experiment.bat`을 더블클릭한다. 터미널에서 창을 멈추지 않고 실행하려면 `run_second_experiment.bat --no-pause`를 사용한다. 실행기는 `xpool\openai\clip-vit-base-patch32` 안의 모델·토크나이저 파일을 확인하고 절대 경로로 전달한다. 파일 배치가 다르면 아래 Python 실행기에 `--clip-model-path`, `--videos-dir`을 지정한다.

배치 32로 **chunk 2와 4 각각 10 updates**를 새 폴더에서 실행한다. 둘 다 GradCache encoder 재계산을 검사하고, 유한 loss, 종료 상태, 실제 기록 수, step 시간과 VRAM을 확인할 수 있다. `outputs/gradcache32_smoke_c{2,4}_<timestamp>/`에 `training_profile.json`, `training_steps.csv`, `gpu_samples.csv`, `monitor_report.json`, 학습 로그와 체크포인트를 기록한다.

PyTorch peak allocated/reserved와 장치 전체 VRAM을 따로 본다. 각 smoke에서 초기 몇 step을 빼고 시간과 여유 VRAM을 비교해, 통과한 설정 중 충분한 여유가 있는 빠른 chunk를 선택한다. Chunk 4는 업데이트당 encoder chunk 8개, chunk 2는 16개다. 27.7시간 추정은 chunk 4 기준이며 첫 실측에서 다시 갱신한다. 32×32 pooling과 손실도 메모리를 쓰지만 CLIP 역전파가 주된 비교 대상이다. 두 설정이 메모리 한도에 걸리면 chunk 1을 별도 실행에서 확인한다.

2026-11-19에 오프라인 배치 파일을 실제 RTX 3060 Ti 8GiB에서 검증했다. 두 설정 모두 재계산 검사와 10 updates를 완료했다. 시간은 초기 2 updates를 뺀 8 updates의 평균이다.

| GradCache chunk | 평균 초/update | PyTorch peak allocated / reserved | 장치 사용량 최대 | 시작부터 종료까지 |
| --- | ---: | ---: | ---: | ---: |
| 2 | 3.92초 | 3,356 / 3,592 MiB | 6,500 MiB | 81.5초 |
| 4 | 3.00초 | 4,329 / 4,660 MiB | 7,568 MiB | 66.6초 |

chunk 4가 이 smoke에서 더 빨랐다. 장치에서는 시작할 때 약 1,550MiB를 사용 중이었고, chunk 4의 피크에서 624MiB가 남았다. 300 step 운영 파일럿에서는 GPU 여유를 확보한 뒤 chunk 4를 우선 확인하고, 비슷한 점유가 계속되면 여유가 큰 chunk 2를 사용한다. 위 수치는 재계산 검사를 켠 짧은 관측이다. 긴 학습 시간은 파일럿의 검사 없는 처리량으로 갱신한다.

같은 실험명으로 GradCache chunk 4는 step 10→11, 누적-32는 step 2→3 재개에 성공했다. 저장된 모델의 유한값과 sampler·optimizer·scheduler 위치를 대조했고, 세 smoke에서 seed 24의 셔플 순열이 일치했다. CPU 테스트 36개도 통과했다. 이 검증의 범위는 smoke와 재개이며, 300 step 파일럿과 5 epoch 학습 결과는 아직 없다.

## 9,600쌍 운영 파일럿

Smoke와 같은 seed로 실행하되 재계산 검사 플래그는 빼고 **새 실험명**을 쓴다. 다음 명령의 chunk 4는 예시다. 파일럿 값은 **속도, VRAM, 안정성** 판단에만 쓴다. 세 실행 모두 9K 전체를 사용하고 `evals_per_epoch=0`이다. 1K-A 추론, R@1이나 raw training loss 비교는 이 단계에 포함하지 않는다.

```powershell
$py = '..\xpool\.venv\Scripts\python.exe'
& $py run_training_session.py --hours 1 --max-steps 300 --exp-name gradcache32_pilot300 --batch-size 32 --grad-cache-chunk-size 4
& $py run_training_session.py --hours 1 --max-steps 300 --exp-name accum32_pilot300 --batch-size 32 --accumulation-chunk-size 4
& $py run_training_session.py --hours 1 --max-steps 2400 --exp-name batch4_pilot2400 --batch-size 4
& $py summarize_second_experiment.py --baseline outputs\batch4_pilot2400 --gradcache outputs\gradcache32_pilot300 --accumulation outputs\accum32_pilot300 --output outputs\second_experiment_comparison.json
```

배치 32의 300 updates와 배치 4의 2,400 updates 모두 각 warmup의 약 10.7% 지점에서 끝난다. 배치 4가 그 구간에 적용한 학습률의 합은 약 8배다. 누적-32와 GradCache-32끼리는 스케줄이 같지만 이 early-warmup 파일럿의 판정값은 운영 지표로 고정한다. 성능 기반 탐색이 추가로 필요하면 별도 학습 8K/내부 검증 1K 분할을 마련한 새 실험에서 수행한다. [기존 8K/1K 분할](results/margin_pilot_20261109/README.md)

| 작업 | 사전 시간 추정 |
| --- | ---: |
| GradCache smoke 두 설정 | 약 2–3분, 로딩·저장 포함 |
| GradCache-32 300 updates, chunk 4 | 약 18분 |
| 누적-32 300 updates / 배치 4 2,400 updates | 각각 약 15분 |
| GradCache-32 5 epochs, chunk 4 | 약 28시간/seed |
| 누적-32 5 epochs | 약 24시간/seed |

각 실행에서 `duplicate_video_batches`는 **effective batch 전체**의 영상 ID 중복, `duplicate_negative_batches`는 **실제로 loss를 구성하는 그룹** 안의 중복이다. 분모는 `video_id_steps`다. 영상당 caption 20개를 무작위로 섞을 때 중복 probability는 배치 32에서 약 5%, 배치 4에서 약 0.06–0.07%다. 누적-32의 chunk 간 중복은 서로의 negative가 되지 않는다. 실제 횟수는 프로파일을 사용하며 세션이 나뉘면 각 세션 수치를 합산한다.

## 분할 실행으로 5 epochs 학습

파일럿 종료·VRAM·finite loss, 동일 설정의 체크포인트 재개를 확인한 뒤 별도 실험명으로 시작한다. 아래는 seed 24의 예다. GradCache의 `4`는 smoke에서 고른 chunk 값으로 지정한다. 시간 예산은 세션마다 조절할 수 있다.

```powershell
$py = '..\xpool\.venv\Scripts\python.exe'
# --dry-run을 붙이면 로컬 경로와 선택될 체크포인트를 미리 확인한다.
& $py run_training_session.py --hours 7 --exp-name gradcache32_full_s24 --batch-size 32 --grad-cache-chunk-size 4 --seed 24
& $py run_training_session.py --hours 7 --exp-name accum32_full_s24 --batch-size 32 --accumulation-chunk-size 4 --seed 24
& $py run_training_session.py --hours 7 --exp-name batch4_full_s24 --batch-size 4 --seed 24
```

세 모드는 한 GPU에서 각각 실행한다. 각 실험의 다음 세션에는 **같은 명령의 실험명·배치·chunk·seed**를 유지하고 필요하면 `--hours`만 바꾼다. 새 폴더에서 처음 시작하면 기본 CLIP으로 초기화하고, 그 뒤에는 해당 폴더의 `checkpoint-last.pth`를 고른다. 중단 기록만 있고 체크포인트가 없으면 명확한 오류를 낸다. 이전 배치 4 가중치는 명시적인 `--resume`을 통해 같은 설정에서만 이어갈 수 있다. 다른 모드나 chunk 크기로 바꿔 재개할 수 없다. 실험 폴더가 다르면 이전 로그는 보존된다. 상세 저장 동작은 [분할 실행 안내](RESUME_TRAINING.md)를 참고한다.

배치 32에서 `global_step=28125`, 배치 4에서 `global_step=225000`에 도달하고 `checkpoint-epoch5.pth`가 끝까지 저장된 것을 확인한 뒤 아래를 각각 한 번 실행한다. 시간 제한 중에는 파일명에 epoch 5가 붙은 **부분 체크포인트**가 있을 수 있으므로 먼저 step 수와 `partial_epoch=false`를 확인한다.

```powershell
$py = '..\xpool\.venv\Scripts\python.exe'
$clip = '..\xpool\openai\clip-vit-base-patch32'
$videos = '..\xpool\data\MSRVTT\vids'
foreach ($run in 'gradcache32_full_s24','accum32_full_s24','batch4_full_s24') {
    $expected = if ($run -like 'batch4_*') { 225000 } else { 28125 }
    $profile = Get-Content "outputs\$run\training_profile.json" -Raw | ConvertFrom-Json
    if ($profile.status -ne 'finished' -or $profile.global_step -ne $expected) {
        throw "Incomplete training: $run"
    }
    & $py test.py --exp_name $run --videos_dir $videos --clip_model_path $clip --huggingface --dataset_name MSRVTT --msrvtt_train_file 9k --load_epoch 5 --batch_size 4 --eval_batch_size 4 --num_workers 2 --no_tensorboard
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $run" }
}
```

Seed 25 반복은 `_s25` 이름과 `--seed 25`로 시작해 같은 절차를 적용한다. 위 비교 조건과 seed를 고정한 뒤 1K-A를 평가하며, 결과를 보고 설정이나 체크포인트를 바꿔 다시 고르는 과정은 두지 않는다.

[GradCache 원 논문](https://arxiv.org/abs/2101.06983) · [First experiment 구현·결과](FIRST_EXPERIMENT.md)
