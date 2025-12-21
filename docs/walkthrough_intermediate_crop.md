# 워크스루 - 중간 크롭 과정 시각화

`A2RL_Batch.py`에 중간 크롭 과정 시각화 기능을 구현했습니다. 이를 통해 강화 학습 과정의 각 단계에서 바운딩 박스가 어떻게 조정되는지 확인할 수 있습니다.

## 변경 사항

### `A2RL_Batch.py`

- **`draw_bbox` 함수 추가**: 이미지에 빨간색 바운딩 박스(두께 2px)를 그립니다.
- **`auto_cropping` 업데이트**: 이제 `image_names`와 `temp_dir`를 인자로 받습니다. 바운딩 박스가 그려진 이미지의 중간 상태를 `temp_dir/<image_name>/step_XXX.jpg`로 저장합니다.
- **`process_batch` 및 `process_single_image` 업데이트**: 시각화를 활성화하기 위해 필요한 인자를 `auto_cropping`에 전달합니다.

## 검증 결과

### 자동화 테스트
에이전트 환경에서 `tf15` conda 환경을 사용하여 스크립트를 성공적으로 실행했습니다.

### 수동 검증
원래 사용하시던 명령어로 변경 사항을 확인할 수 있습니다:

```bash
conda activate tf15
python A2RL_Batch.py --mode directory --input_dir ./images/ --output_dir ./cropped_verification/ --batch_size 2 --verbose
```

실행 후 `cropped_verification_tf15_v2/cropped_temp/` 디렉토리에 각 이미지별 폴더가 생성되고, 단계별 이미지(`step_XXX.jpg`)가 정상적으로 저장된 것을 확인했습니다.
