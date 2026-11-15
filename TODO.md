# TODO — First Experiment

지금은 **긴 학습 없이 3가지 사전 진단만 실행**한다. 상세 설명: [FIRST_EXPERIMENT.md](FIRST_EXPERIMENT.md).

1~3은 2026-11-15에 Windows와 RTX 3060 Ti에서 실행을 마쳤다(총 6분 24초). 결과: [results/first_experiment_20261115](results/first_experiment_20261115/README.md). `xpool` 폴더가 이 저장소 옆에 있으면 `run_first_experiment.bat`을 더블클릭해 세 명령을 한 번에 다시 돌릴 수 있다.

- [x] **1. GradCache gradient 검증** (실제 MSR-VTT + GPU)
  ```powershell
  .\.venv\Scripts\python.exe first_experiment.py gradcheck
  ```
  - `GRADCHECK PASSED`, full vs. 2+2 gradient 상대 오차, feature 재계산 오차, peak VRAM 확인
  - 결과: `gradcheck_report.json`
  - **완료**: PASSED. 상대 오차 8.1e-4~1.3e-3(허용치 1e-2), 재인코딩 차이 0.0, peak 2170MiB(GradCache) / 3133MiB(full)

- [x] **2. 학습 step 병목 측정**
  ```powershell
  .\.venv\Scripts\python.exe first_experiment.py steptime
  ```
  - `data_wait` vs. `gpu_step` 비중 확인 → 프레임 사전 추출 필요 여부 판단
  - 결과: `steptime_report.json` (`step_times.csv`, `decode_times.csv`)
  - **완료**: gpu-bound. `data_wait` 1.4%, `gpu_step` 93.8%. GradCache-32 5 epoch 추정 27.7h(기존 23.9h)

- [x] **3. Feature drift 측정**
  ```powershell
  .\.venv\Scripts\python.exe first_experiment.py drift
  ```
  - 50~600 step의 text/frame feature 변화, stale logit error, 권장 bank 크기 확인
  - 결과: `drift_report.json`, `drift_by_lag.csv`
  - **완료**: 권장 bank 0. lag 50 step(200개)에서 이미 오차 비율 0.141 > 0.1(peak LR). 50 step 미만은 측정하지 않음

- [ ] **4. 결과 공유 및 다음 실험 결정**
  - `outputs/first_experiment/` 아래의 위 JSON/CSV를 모아 비교 → [results/first_experiment_20261115](results/first_experiment_20261115/README.md)에 정리함
  - **GradCache 32 진행 가능 여부 / 디코딩 최적화 필요 여부 / XBM bank 크기** 결정

> 실행 결과가 나오기 전에는 22시간짜리 전체 학습이나 Memory Bank 확장 구현을 시작하지 않는다.
