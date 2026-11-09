# X-Pool Experiment on RTX 3060 Ti

RTX 3060 Ti 8GiB에서 X-Pool의 공식 평가 재현, VRAM 측정, 시간 제한 학습과 재개를 검증한 실험 저장소입니다. 실험 기간은 2026년 11월 7–9일입니다.

**공식 체크포인트의 Recall@1 46.9%를 재현했고, 배치 4로 총 225,000 step의 학습 스케줄을 완료했습니다. 직접 학습한 최종 모델의 Recall@1은 39.9%입니다.** 두 결과는 학습 조건이 다르므로 구분해서 사용해야 합니다.

| MSR-VTT 1K-A, text-to-video | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| 공식 9K 체크포인트 평가 | 46.9% | 72.8% | 82.2% |
| 직접 학습한 배치 4 최종 모델 | 39.9% | 67.7% | 77.7% |

학습 누적 시간은 약 **21시간 55분**, 장치 전체 VRAM 최대 사용량은 **6.59GiB**였으며 메모리 오류는 관측되지 않았습니다.

- [상세 실험 보고서와 다음 실험 계획](report.md)
- [최종 결과 JSON](results/msrvtt9k_split_training/completion_summary.json)
- [최종 체크포인트 검증 기록](results/msrvtt9k_split_training/checkpoint_verification.json)
- [분할 실행 설명](RESUME_TRAINING.md)
- [전체 공개 기록 목록과 SHA-256](results/artifact_manifest.json)

## 코드 출처

기반 코드는 [layer6ai-labs/xpool](https://github.com/layer6ai-labs/xpool)의 커밋 [`bc88718199007d395136adc99a3e69adbee874a0`](https://github.com/layer6ai-labs/xpool/tree/bc88718199007d395136adc99a3e69adbee874a0)입니다. 원 논문은 **X-Pool: Cross-Modal Language-Video Attention for Text-Video Retrieval, CVPR 2022**입니다. 저자, 논문 인용 형식과 원본 사용법은 [원본 README](README_UPSTREAM.md)에 보존했습니다. 이 저장소는 해당 구현을 바탕으로 수행한 별도 실험 기록입니다.

추가 코드는 시간 제한, 측정, 부분 epoch 재개, 셔플·난수 상태 저장과 회귀 테스트를 제공합니다. 모델 구조를 개선한 실험은 아직 수행하지 않았습니다.

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

새 학습, 중단 후 재개, 최종 모델 평가 명령은 [보고서의 재현 절차](report.md#재현-절차)에 있습니다. 분할 실행기는 기존 체크포인트가 필요하므로 빈 저장소에서 곧바로 `run_training_session.py`를 실행하지 말고 먼저 새 학습을 시작해야 합니다.

## 공개 파일 범위

코드, 기존 split·annotation, 실행 설정, 전체 step CSV, GPU 측정 CSV, 로그, 요약 JSON, 초기 6시간 VRAM 그래프를 포함합니다. 영상, 가상환경, 다운로드한 CLIP 가중치와 학습 체크포인트는 포함하지 않습니다. 최종 체크포인트의 해시는 검증 기록에 남아 있으며, 가중치 파일은 원래 실험 PC에 보관되어 있습니다.

`results/`는 과거 실행 기록의 공개 사본입니다. JSON 내부의 `outputs/...`는 당시 실행 경로를 뜻하며 공개 사본은 `results/...`에 있습니다. 개인 PC의 절대 경로는 `<repo>` 또는 `<user-home>`으로 치환했고, 수치와 CSV는 유지했습니다. 원본·공개 사본의 해시는 artifact manifest에서 확인할 수 있습니다.

## 재개 코드 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

기존 실제 GPU 재개 비교 결과는 [resume_gpu_verification.json](results/resume_gpu_verification.json)에 있습니다. 공개용 복사본에서 다시 실행한 CPU 회귀 테스트 기록은 `results/validation/`에 보관합니다.
