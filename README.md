# Capstone Design2 [경희대학교 컴퓨터공학과]

## 자율주행 카트에서의 엣지클라우드 프로토타입 구축
<img src = "./Resources/concept.png">

카트 상의 이미지 자체를 Core Cloud서버로 전송할 경우, 네트워크 부하가 많이 발생하게 된다. 

이에따라, Cloude-Edge단에서 원본 프레임을 머신러닝 모델의 input형식에 맞추어 전처리 작업을 진행한다. 이후 전처리된 데이터만을 Core Cloud에 송신하며, Core Cloud는 사전에 로드된 머신러닝 모델을 통해 예측값을 반환한다. 이후, Edge-Cloud는 결과값만을 수신하게된다. 

이를통해 최종적으로 네트워크의 부하를 분산하여 고성능의 컴퓨팅환경을 구축한다. 최종적으로 시간대비 영상 처리모듈의 퍼포먼스 향상, 기존 연구의 병목현상 해소를 목표한다.

- 지도교수님 : 허의남


### 진행상황
- AWS DeepLens 관련 라이브러리 조사 및 Cloud 서비스 구상. [09.17]

- 주제 선정 : 자율주행 카트에서의 엣지클라우드 프로토타입 구축-영상처리모듈 [09.21]

### Revision History
- 20.09.17 면담확인서 업로드
- 20.09.21 주제 선정 및 슈도 코드 작성, 주간보고서 업로드
