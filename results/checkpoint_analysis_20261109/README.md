# 체크포인트 비교와 검색 질의 분석

2026-11-09에 보존된 모델 7개를 MSR-VTT 1K-A의 동일한 1,000개 질의에서 추가 평가했다. 새 학습은 수행하지 않았다.

자체 모델 중 관측된 최고 R@1은 **epoch 4의 40.3%**였으며 공식 모델 46.9%와 6.6%p 차이가 남았다. 공식 모델만 맞힌 117건 중 100건은 epoch 4에서 정답이 2–5위였다. Attention은 자체 모델에서 더 균등했지만, 회복 사례에서도 같은 경향이 있어 성능 저하의 원인으로 단정하지 않았다.

- [전체 실험 보고서](../../report.md)
- [모델별 성능 CSV](metrics_summary.csv)
- [모든 질의의 paired rank·top-1 비교](rank_comparison.csv)
- [집계 JSON](analysis_summary.json)
- [표본 프레임을 관찰한 실패·회복 사례 7건](casebook.md)
- [결과 파일 해시 목록](artifact_manifest.json)

## 보존 모델과 파일 경로

| 폴더 | 로컬 체크포인트 | Global step |
| --- | --- | ---: |
| `official/` | `outputs/msrvtt9k/model_best.pth` | 공식 제공 모델 |
| `epoch1/` | `outputs/msrvtt9k_train_vram_20261107_054714/checkpoint-epoch1.pth` | 45,000 |
| `legacy6h/` | `outputs/msrvtt9k_train_vram_20261107_054714/checkpoint-epoch2.pth` | 62,216 |
| `epoch2/` | `outputs/msrvtt9k_split_training/checkpoint-epoch2.pth` | 90,000 |
| `epoch3/` | `outputs/msrvtt9k_split_training/checkpoint-epoch3.pth` | 135,000 |
| `epoch4/` | `outputs/msrvtt9k_split_training/checkpoint-epoch4.pth` | 180,000 |
| `epoch5/` | `outputs/msrvtt9k_split_training/checkpoint-epoch5.pth` | 225,000 |

`legacy6h`는 epoch 2의 중간 snapshot이고 `epoch2`는 이후 epoch 2를 완료한 모델이다. 파일 이름이 같아도 서로 다른 체크포인트이며 각 `evaluation.json`에 SHA-256과 저장된 진행 상태를 기록했다.

각 폴더에는 `evaluation.json`, `evaluation.log`, `queries.csv`, `gpu_samples.csv`가 있다. [run_summary.json](run_summary.json)은 최초 5개 평가, [early_history_run_summary.json](early_history_run_summary.json)은 초기 이력의 추가 2개 평가를 기록한다. 프로세스 시간 합계는 약 17분 6초이며 분석 작업 시간은 별도다.

## 질의별 CSV와 attention

`queries.csv`는 1-based 정답 rank, 정답·top-1 cosine score, top-10 영상 ID·점수, 정답·top-1 영상의 프레임 attention, normalized entropy와 최대 가중치를 포함한다. `top1_caption`은 해당 후보 영상에 연결된 1K-A 평가 caption이며 영상 전체에 대한 검증된 설명은 아니다.

공식 `test.py`와 `t2v_metrics`를 실행하면서 결과를 관찰했다. softmax observer는 기존 출력 tensor를 반환하며 모델 forward와 rank 계산을 대체하지 않는다. 12프레임의 softmax 값은 `[video, head, frame, query]` 축으로 관찰하고 head 평균을 CSV로 내보냈다. 실제 설정의 head 수는 1이다.

평가 당시 코드의 SHA-256은 모든 `evaluation.json`에 있으며, 해당 원문을 [evaluator_used.py.txt](validation/evaluator_used.py.txt)에 보존했다. 이후 공개 CLI에는 기존 결과 폴더를 덮어쓰지 못하도록 사전 검사를 추가했다. 이 보완은 모델의 forward나 metric을 변경하지 않는다.

공식 모델과 epoch 5의 Recall·MedR·MeanR이 이전 평가와 일치했고, 각 모델에서 rank로 다시 계산한 metric도 일치했다. [분석용 CPU 테스트](validation/tests.json)는 rank·attention 축, 동률 처리와 기존 결과 보존을 검사한다. 전체 similarity 행렬은 저장하지 않았으며, 질의별 정답 rank·상위 10개 후보와 attention을 보존했다.

이번 추가 평가의 장치 VRAM 관측 최대는 2574MiB(약 2.51GiB)였다. 2초 간격 측정이며 데스크톱 등 다른 사용량을 포함한다. 현재 loader는 체크포인트를 CPU로 읽는다. 과거 평가의 VRAM과 측정·로딩 조건이 같다고 가정하거나 모델 경량화 효과로 해석하지 않는다.

## 동일 절차 실행하기

`--source-root`에는 영상, 로컬 CLIP 모델·토크나이저, 위 체크포인트를 갖춘 기존 실험 폴더를 지정한다. 아래 `outputs/checkpoint_analysis_recheck`는 새 결과 폴더 예시다. 기존 공개 결과와 같은 경로를 지정하지 않는다.

먼저 epoch 2·3·4, 공식 모델, epoch 5를 평가한다. Python은 기존 PyTorch 1.8.1 환경을 사용한다.

```powershell
.\.venv\Scripts\python.exe analyze_retrieval.py run `
  --source-root ..\xpool --results-dir outputs/checkpoint_analysis_recheck
```

초기 epoch 1과 재개 전환 직전 snapshot도 같은 평가 함수를 직접 호출한다.

```powershell
.\.venv\Scripts\python.exe analyze_retrieval.py evaluate `
  --source-root ..\xpool --results-dir outputs/checkpoint_analysis_recheck `
  --label epoch1 --experiment msrvtt9k_train_vram_20261107_054714 --epoch 1
.\.venv\Scripts\python.exe analyze_retrieval.py evaluate `
  --source-root ..\xpool --results-dir outputs/checkpoint_analysis_recheck `
  --label legacy6h --experiment msrvtt9k_train_vram_20261107_054714 --epoch 2
```

집계는 표준 라이브러리만으로 가능하다. 그림은 [plotting_environment.json](plotting_environment.json)에 기록한 별도 Python 3.11.4 / matplotlib 3.10.1 / NumPy 2.2.4 환경에서 생성했다. 아래 `python`은 해당 plotting 환경의 Python을 뜻한다.

```powershell
python summarize_retrieval.py --results-dir outputs/checkpoint_analysis_recheck --plots
```

영상 프레임의 로컬 검토 그림은 기존 평가 환경에서 생성한다. 기본 출력은 Git에서 제외한 `.cache/checkpoint_analysis_frames/`다. 실제 평가의 resize·center crop을 사용하고 표시할 때만 CLIP 정규화를 되돌린다.

```powershell
.\.venv\Scripts\python.exe render_analysis_frames.py `
  --source-root ..\xpool --results-dir outputs/checkpoint_analysis_recheck
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_retrieval_analysis.py -v
```

## 해석의 범위

epoch 4의 최고값은 같은 1K-A 테스트셋에서 비교한 관찰 결과다. 독립 validation에서 선택한 모델이나 새 일반화 성능으로 주장하지 않는다. 새로운 학습 조건·구조를 고르려면 학습 데이터 내부의 별도 validation을 사용해야 한다.

정성 사례는 7건의 rank 기반 선택이며, 전체 실패 유형의 빈도를 추정하는 표본이 아니다. 원본 영상이나 추출 프레임은 공개하지 않았고, 그래프와 재현용 index·ID만 포함했다. Attention 관찰은 인과 분석이나 attention 변경의 성능 개선 증거가 아니다.
