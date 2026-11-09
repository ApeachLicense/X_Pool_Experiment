# X-Pool 학습을 두 번에 나누어 실행하기

PowerShell에서 이 파일이 있는 `xpool` 폴더로 이동한 뒤 실행합니다.

2026년 11월 8일에는 최대 7시간 실행합니다.

```powershell
.\.venv\Scripts\python.exe run_training_session.py --hours 7
```

2026년 11월 9일에는 저장된 위치부터 최대 10시간 실행합니다.

```powershell
.\.venv\Scripts\python.exe run_training_session.py --hours 10
```

시간은 각 실행의 학습 시작 시점부터 새로 계산합니다. 총 목표는 기존 실험과 같은 5 epoch입니다. 5 epoch를 마치면 시간 예산이 남아 있어도 종료합니다. 마지막 optimizer 업데이트와 체크포인트 저장 때문에 설정 시간보다 몇 초 더 걸릴 수 있습니다.

기본 설정은 MSR-VTT 9K, 배치 4, FP32, 12프레임, 224×224, 데이터 작업자 2개입니다. 학습 중 평가는 꺼져 있습니다. GPU 점유와 온도는 5초 간격으로 기록합니다.

첫 실행은 기존 6시간 실험의 `outputs/msrvtt9k_train_vram_20261107_054714/checkpoint-epoch2.pth`에서 62,216스텝과 optimizer·학습률을 복원합니다. 이후 실행은 `outputs/msrvtt9k_split_training/checkpoint-last.pth`를 자동으로 선택합니다. 다른 파일을 지정할 때는 `--resume 경로`를 추가합니다.

기존 6시간 체크포인트에는 당시 셔플과 난수 상태가 없습니다. 따라서 첫 재개부터 seed 24에 따른 epoch별 셔플과 샘플별 증강 규칙을 적용합니다. 중단된 2번째 epoch의 17,216개 완료 배치 다음 위치부터 진행하며, 첫 재개 이후 데이터 순서는 이전 6시간 실행의 순서와 달라질 수 있습니다.

새 체크포인트에는 다음 배치 위치, 실제 셔플 순열, Python·NumPy·PyTorch CPU·CUDA 난수 상태, 모델·optimizer·학습률 scheduler를 저장합니다. 데이터 로더가 미리 읽은 배치는 완료 위치에 포함하지 않습니다. 동일 환경에서 이후 세션을 재개하면 다음 샘플과 dropout·증강을 이어갈 수 있습니다.

최신 요약은 실험 폴더의 `training_profile.json`과 `monitor_report.json`에 있습니다. 이전 실행의 로그와 CSV는 유지하며, 추가 실행의 기록은 해당 폴더 아래 `sessions`에 저장됩니다.

실행 예정 명령과 체크포인트 경로만 확인할 때는 다음 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe run_training_session.py --hours 7 --dry-run
```

재개 검증은 다음 명령으로 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

실제 X-Pool GPU에서도 4스텝 연속 실행과 2스텝 뒤 재개한 2스텝 실행의 모델 가중치, optimizer, scheduler, 난수와 셔플 상태가 모두 비트 단위로 일치했습니다. 이 비교 결과는 `outputs/resume_gpu_verification.json`에 있습니다.
