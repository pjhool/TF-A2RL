# Git Issue Draft

**제목:** [Feature] A2RL 배치 처리를 위한 중간 크롭 시각화 기능 추가

## 배경 (Background)
강화 학습 모델이 이미지를 크롭하는 과정에서, 각 스텝마다 바운딩 박스가 어떻게 이동하고 변화하는지 시각적으로 확인하기 어렵습니다. 디버깅 및 모델 동작 이해를 돕기 위해 중간 과정을 이미지로 저장하는 기능이 필요합니다.

## 변경 사항 (Changes)
`A2RL_Batch.py` 스크립트에 다음 기능을 추가했습니다:
1.  **`draw_bbox` 함수**: 이미지 위에 빨간색(Red) 바운딩 박스를 그리는 유틸리티 함수 구현.
2.  **중간 저장 로직**: `auto_cropping` 함수 내에서 매 스텝(step)마다 바운딩 박스가 그려진 이미지를 `cropped_temp/<파일명>/` 디렉토리에 저장하도록 수정.
3.  **인자 전달**: `process_batch` 및 `process_single_image` 함수에서 파일명과 임시 디렉토리 경로를 전달하도록 수정.

## 검증 결과 (Verification)
- **환경**: Conda `tf15` 환경
- **테스트 커맨드**:
  ```bash
  python A2RL_Batch.py --mode directory --input_dir ./images/ --output_dir ./cropped_verification/ --batch_size 2 --verbose
  ```
- **결과**:
  - `cropped_verification/cropped_temp/` 폴더 생성 확인.
  - 각 원본 이미지 이름으로 된 하위 폴더 생성 확인.
  - 하위 폴더 내에 `step_001.jpg`, `step_002.jpg` 등 단계별 시각화 이미지 생성 확인.

## 관련 문서 (References)
- 상세 워크스루: `docs/walkthrough_intermediate_crop.md`
