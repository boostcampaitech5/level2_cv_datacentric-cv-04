# [level2-cv-04]_의료비 영수증 OCR 
- Project Period 2023/05/02 ~ 2023/05/18
- Project Wrap-Up Report (https://docs.google.com/document/d/1B5JuvBl3AQFBHdLms5SJef3XLPhYCU_N10V6ybgzdZM/edit?usp=sharing)

## ✏️ Project Overview

- 데이터: 다양한 크기와 형태의 의료비 영수증 301장  
- 프로젝트 주제: 다수의 노이즈가 있는 의료비 영수증 사진에서의 글자검출 
- 평가지표: f1- score, recall, precision
- 활용 장비 및 재료
  - 컴퓨팅 환경: Nvidia V100 GPU (총 5대)
  - 협업 및 실험관리 툴: notion, git, slack, jira, wandb

## 🙌 Members

| 강동화 | 박준서 | 서지희 | 장철호 | 한나영 |
| :---: | :---: | :---: | :---: | :---: |
| <img src = "https://user-images.githubusercontent.com/98503567/235584352-e7b0568f-3699-4b6e-869f-cc675631d74c.png" width="120" height="120"> | <img src = "https://user-images.githubusercontent.com/89245460/234033594-cb90a3c0-f0dc-4218-9e11-2abc8db2be67.png" width="120" height="120"> |<img src = "https://user-images.githubusercontent.com/76798969/234210787-18a54ddb-ae13-4554-960e-6bd45d7905fb.png" width="120" height="120">  | <img src = "https://avatars.githubusercontent.com/u/70846128?s=400&u=6309e4d3b06e87d1a400f130efb6d6b5d6198f7d&v=4" width="120" height="120" /> |<img src = "https://user-images.githubusercontent.com/76798969/233944944-7ff16045-a005-4e4e-bf59-632766194d7f.png" width="120" height="120" />|
| [@oktaylor](https://github.com/oktaylor) | [@Pjunn](https://github.com/Pjunn) | [@muyaaho](https://github.com/muyaaho) | [@JCH1410](https://github.com/JCH1410) | [@Bandi120424](https://github.com/Bandi120424) |


## 🌏 Contributions

| 팀원명 | 작업 |
| :---: | :---: |
| 강동화 | EDA, GitHub 환경 세팅, Augmentation, Annotation tool research|
| 박준서 | EDA, Augmentation 리서치 및 실험, streamlit을 사용하여 모델 평가를 위한 웹사이트 구현 |
| 서지희 | EDA, 외부 data 조사, 금융 OCR 데이터 추가, fine tuning 진행 |
| 장철호 | EDA, Augmentation 리서치 및 실험|
| 한나영 | EDA, 세부 평가 지표 설정, 외부 data 및 검증 데이터 구성, Augmentation 리서치 및 실험 |

## :scroll: 프로젝트 수행 결과
### 검증 데이터 구성 
- 스캔 여부 및 가로 세로 비율에 따라 그룹화 → 각 그룹의 비율을 고려하여 훈련/검증 데이터 구성

| | 세로/가로 < 1, 스캔 X  | 세로/가로 >= 1, 스캔 X  | 세로/가로 < 1, 스캔 O | 세로/가로 >= 1, 스캔 O |
| :---: | :---: | :---: | :---: | :---: |
| train set 비율| 2.07% | 39.83% |7.47% | 50.62% |
| val. set 비율| 1.67% | 40% | 6.67% | 51.67% |

### 실험 결과 
- 다양한 augmentation 기법을 활용하여 test 환경과 유사한 형태를 가질 수 있도록 함
- default augmentation: resize(size=2048), adjust_height(ratio=0.2), rorate_img(범위: ±10°), crop_img(size=1024), ColorJitter(0.5, 0.5, 0.5, 0.25), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

| No. | 추가 augmentation  | F1 score  | 비고 |
| :---: | :---: | :---: | :---: |
| 1 | ShadowCast | 0.9463 | 외부데이터로 추가 학습 |
| 2 | ShadowCast | 0.9515 | - |
| 3 | ShadowCast + Blur(blur_limit=5,p=0.25) | 0.9326 | - |
| 4 | ShadowCast + PixelDropout(dropout_prob = 0.05, p=1) | 0.9422 | - |
| 5 | 0.8의 확률로 RandomRain, PixelDropout 중 하나 적용 | 0.9329 | pickle 이용 (이미지의 같은 부분을 학습) |
| 6 | 0.8의 확률로 RandomRain(color=black, white), PixelDropout 중 하나 적용 | 0.9327 | pickle 이용 (이미지의 같은 부분을 학습) |
| 7 | ColorJitter = False + GaussNoise(p=0.5, var_limit=(10.0, 50.0))| 0.9622 | - |
| 8 | ColorJitter = False + RandomBrightnessContrast(brightness_limit=(0, 0.25), contrast_limit=(-0.5, 0), p=0.5))| 0.9252 | pickle 이용 (이미지의 같은 부분을 학습) |
| 9 | ColorJitter = False + RandomBrightnessContrast(brightness_limit=(0, 0.25), contrast_limit=(-0.5, 0), p=0.8))| 0.9568 | pickle 이용 (이미지의 같은 부분을 학습) |





