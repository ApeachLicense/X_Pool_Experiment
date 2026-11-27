# AGENTS.md — X-Pool 연구 및 실험 지침

이 지침은 저장소 전체에 적용한다. 현재 작업은 RTX 3060 Ti 8GiB에서 GradCache-32로 재현한 X-Pool을 바탕으로 one-to-many 문제를 연구하는 것이다. 단순 성능 경쟁이나 최신 텍스트–영상 검색 모델로 교체하는 것이 아니라, 개별 방법론의 효과를 재현 가능하게 검증하는 것이 목적이다.

## 1. 작업 시작 및 범위

- 먼저 `README.md`, `TODO.md`, `RESUME_TRAINING.md`를 읽고 현재 branch, HEAD, working tree 및 기존 결과를 확인한다.
- `AGENTS.md`는 공통 연구 원칙, `README.md`는 구현/실행/평가 프로토콜, `TODO.md`는 현재 작업과 후속 연구 목록, `RESUME_TRAINING.md`는 체크포인트 재개 절차의 기준이다.
- 사용자가 지정한 방법론만 진행한다. 현재 `third`의 활성 과제는 TODO 01 SupCon 검증 및 실험이다. TODO 02–05는 후속 연구이며 별도 요청 없이 구현하거나 학습하지 않는다.
- 사용자가 지정한 브랜치에서만 변경한다. `main`이나 기존 백업 브랜치를 수정·병합하지 않는다. 사용자의 기존 변경사항을 되돌리거나 덮어쓰지 않는다. 커밋/푸시는 요청된 범위에서만 수행한다.
- 문서와 코드, 실제 실험 기록이 충돌하면 임의로 설정을 변경하지 말고 근거와 불일치를 보고한다. 명령을 실행하기 전에 해당 코드의 실제 옵션을 확인한다.

## 2. 연구 및 구현 원칙

- 원 논문의 문제 정의, 목적함수, 가정, 알고리즘을 확인하고 X-Pool에 적용한 변형과 원 논문 그대로의 구현을 구분한다.
- 기본 모델의 CLIP encoder, text-conditioned pooling, cosine retrieval score, GradCache의 캐싱 위치는 해당 실험의 연구 대상이 아닌 한 유지한다.
- 기존 `--loss clip` 및 학습/평가/재개 경로를 보존한다. 새로운 목적함수는 명시적인 옵션으로 선택할 수 있게 구현한다.
- 동일 seed의 baseline과 candidate는 데이터 split, pair 순서, augmentation, 초기화, optimizer, learning rate, scheduler, 업데이트 예산 및 평가 방식이 같아야 한다. 변경 변수는 원칙적으로 하나로 제한한다.
- sampler, hard-negative 정책, 다른 loss, 모델 구조, precision 또는 학습 예산 변경은 별도 ablation으로 구분한다. 사전에 변경 이유와 예상 영향을 문서화하고, 기존 비교와 섞지 않는다.
- 기존 환경(Windows, Python 3.9.18, PyTorch 1.8.1+cu111)과 호환되도록 작성한다. 불필요한 의존성, GPU 메모리 증가, 대형 리팩터링을 피한다.

## 3. 기본 실험 프로토콜

별도 사용자 지시가 없다면 아래 조건을 기본으로 사용한다. 실행 명령은 README의 해당 방법론 절을 참조한다.

| 항목 | 기본값 |
| --- | --- |
| GPU | RTX 3060 Ti 8GiB |
| Dataset | MSR-VTT 9K 학습 |
| 최종 평가 | MSR-VTT 1K-A, 평가 배치 4 |
| Effective batch / GradCache chunk | 32 / 4 |
| Seed | 24, 25 (각각 독립 실행) |
| Epochs / optimizer updates | 5 / seed당 28,125 (9K 전체 학습) |
| CLIP / pooling LR | 1e-6 / 3e-5 |
| 초기화 | 동일 사전학습 CLIP 및 새 X-Pool pooling |
| Optimizer / weight decay | 기존 AdamW / 0.2 |
| LR schedule | 기존 warmup 0.1 및 cosine schedule |
| Precision | 기존 FP32 설정 유지 |
| 입력 | 12 uniform frames, 224×224 |
| 세션 | 기본 7시간, 동일 명령으로 재개 |

내부 검증셋을 쓰는 파일럿은 별도 프로토콜로 정의한다. 8K/1K 내부 분할에서는 9K 전체를 학습한 체크포인트로 초기화하지 않는다. 분할, 예산, 판정 기준을 시작 전에 고정한다.

## 4. 검증 게이트 및 실행 순서

1. 구현 검토: 수식/shape/positive 및 negative 정의/수치 안정성/gradient 경로를 확인한다.
2. CPU 검증: 해당 loss의 단위 테스트, 기존 동작과의 회귀 테스트, full-batch 대비 GradCache gradient 동등성 및 재개 호환성을 확인한다. 중복 positive가 microbatch 경계를 넘는 경우, singleton 및 마지막 짧은 chunk도 검증한다.
3. GPU smoke test: 실제 데이터/CLIP으로 README에 명시된 짧은 학습을 실행한다. SupCon은 10 optimizer updates, batch 32/chunk 4 및 feature replay 검사로 검증한다. 정상 종료, finite loss, 재계산 검사, VRAM/OOM, checkpoint 및 로그를 확인한다.
4. 사전 진단: SupCon의 경우 seed별 배치 내 중복 video ID 빈도와 실제 multi-positive 적용 범위를 기록한다. 중복이 없으면 원 CLIP loss와 동일하다는 한계를 명시한다.
5. 본 학습: 앞 단계가 통과한 경우에만 seed 24, 25를 각각 새 실험 이름으로 시작한다. 한 장의 GPU에서 동시에 실행하지 않는다.
6. 최종 평가: 각 seed의 5번째 epoch 완료 및 global_step=28125, partial_epoch=false를 확인한 뒤 동일 평가 방식으로 평가한다.
7. 기록: 실측 결과, 비교, 한계 및 후속 작업을 정리하고 README/TODO를 최신화한다.

테스트가 실패하면 원인을 진단하고 최소 변경으로 수정한 뒤 해당 검증을 다시 실행한다. 실패 상태에서 본 학습으로 넘어가지 않는다. 실행 환경이나 시간/데이터가 부족하면 완료한 단계, 미실행 단계, 재개 명령과 차단 원인을 명시하고 멈춘다. 테스트 코드 작성과 테스트 통과를 구분하며 실행하지 않은 결과를 통과로 표시하지 않는다.

## 5. 체크포인트 및 GPU 관리

- 새 방법론/seed/대조군/smoke test마다 서로 다른 experiment name을 사용한다.
- 본 학습은 새 초기화로 시작한다. smoke test, 다른 seed, 다른 loss의 checkpoint를 resume 용도로 사용하지 않는다.
- 재개는 기존 실험과 동일한 설정 및 loss를 사용해 `checkpoint-last.pth`에서 이어간다. 기존 optimizer/scheduler/RNG/sampler 진행 상태를 보존한다.
- 실행 전 GPU 여유 메모리, 영상/CLIP 경로, 기존 실행 여부를 확인한다. 기존 로그·checkpoint·결과 파일을 덮어쓰거나 삭제하지 않는다.
- OOM 또는 중단 시 로그와 checkpoint를 보존하고 원인을 기록한다. batch size, chunk, precision을 조용히 변경해 다른 실험으로 이어가지 않는다.

## 6. 평가 및 결과 해석

- 공식 1K-A는 최종 평가에만 사용한다. 하이퍼파라미터 선택, checkpoint 선택, 반복적인 방법론 조정에는 내부 검증셋을 사용한다.
- 주 평가 지표는 text-to-video R@1이며 R@5, R@10도 보고한다. 동일 seed의 기존 GradCache-32와 비교하고 seed별 차이와 평균 차이를 %p 단위로 기록한다.
- 기존 GradCache-32의 참고 결과는 seed 24 R@1/5/10 = 45.4/72.5/81.6%, seed 25 = 46.1/72.9/82.0%이다. 기존 실험 조건의 일치 여부를 확인한 후 비교한다. 공식 X-Pool 가중치의 46.9/72.8/82.2%는 참고값이며 통제 대조군이 아니다.
- 개선 기준은 실험 전에 명시한다. 두 seed의 차이와 평균 개선을 보고하되, 두 seed만으로 통계적 유의성이나 일반화를 단정하지 않는다.
- 손실값은 목적함수가 다르면 직접 비교하지 않는다. 성능 하락, 무효 결과, 실패도 숨기지 않는다. 1K-A의 라벨 불완전성 및 의미상 관련된 비라벨 영상과의 차이를 고려한다.

## 7. 재현성과 문서 업데이트

실험마다 `results/<experiment_name>/` 또는 기존 기록 규칙에 맞는 새 경로에 다음을 남긴다.

- 실험 가설, 원 논문/적용 수식, 코드 commit SHA, 실행 명령과 설정, seed, split 및 초기화 정보.
- 실제 CPU/GPU 테스트 명령, 결과, exit code, smoke test 및 resume 확인 기록.
- 학습 로그, step CSV, GPU peak/학습 시간, 중복 video ID 통계 및 checkpoint 식별 정보.
- seed별 최종 평가 결과와 baseline 대비 차이, 결과 해석 및 한계.

기존 `outputs/`의 원본 기록은 보존한다. 대용량 영상/모델 가중치/가상환경 및 비밀 정보는 Git에 올리지 않는다. 공개 결과는 필요한 로그/요약/해시를 남기고 개인 경로를 정리한다.

실제 완료된 작업만 TODO 체크박스를 갱신한다. README의 결과표는 실제 평가 결과가 있을 때만 수정하고, 미측정 값은 그대로 남긴다. 이후 에이전트가 문서만 읽고 다음 단계를 재현할 수 있도록 실행 조건과 미완료 작업을 명시한다.
