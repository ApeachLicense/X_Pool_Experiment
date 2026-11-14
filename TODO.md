# TODO — First Experiment

지금은 **긴 학습 없이 3가지 사전 진단만 실행**한다. 상세 설명: [FIRST_EXPERIMENT.md](FIRST_EXPERIMENT.md).

- [ ] **1. GradCache gradient 검증** (실제 MSR-VTT + GPU)
  ```powershell
  .\.venv\Scripts\python.exe first_experiment.py gradcheck
  ```
  - `GRADCHECK PASSED`, full vs. 2+2 gradient 상대 오차, feature 재계산 오차, peak VRAM 확인
  - 결과: `gradcheck_report.json`

- [ ] **2. 학습 step 병목 측정**
  ```powershell
  .\.venv\Scripts\python.exe first_experiment.py steptime
  ```
  - `data_wait` vs. `gpu_step` 비중 확인 → 프레임 사전 추출 필요 여부 판단
  - 결과: `steptime_report.json` (`step_times.csv`, `decode_times.csv`)

- [ ] **3. Feature drift 측정**
  ```powershell
  .\.venv\Scripts\python.exe first_experiment.py drift
  ```
  - 50~600 step의 text/frame feature 변화, stale logit error, 권장 bank 크기 확인
  - 결과: `drift_report.json`, `drift_by_lag.csv`

- [ ] **4. 결과 공유 및 다음 실험 결정**
  - `outputs/first_experiment/` 아래의 위 JSON/CSV를 모아 비교
  - **GradCache 32 진행 가능 여부 / 디코딩 최적화 필요 여부 / XBM bank 크기** 결정

> 실행 결과가 나오기 전에는 22시간짜리 전체 학습이나 Memory Bank 확장 구현을 시작하지 않는다.
