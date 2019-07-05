## 정치인 뉴스 분석

`latest update`: 2019. 7. 5

이 분석 예시는 [politician_news_dataset][politician_news_dataset] 을 이용합니다. 데이터를 이용할 수 있는 파이썬 코드는 git clone 을 하면 이용할 수 있으며, 실제 데이터는 fetch 함수를 이용하여 다운로드 받아야 합니다. 자세한 사용법은 해당 데이터셋의 README 를 참고하세요.

이 데이터셋은 네이버 뉴스에서 2013. 1. 1 부터 2019. 3. 10 까지 생성된 뉴스를 수집한 데이터입니다. 총 20 개의 카테고리로 이뤄져 있으며, 각 카테고리는 아래의 질의어로 검색된 뉴스로 구성되어 있습니다.

```
category 0: 김무성
category 1: 김영삼
category 2: 나경원
category 3: 노무현
category 4: 노회찬
category 5: 문재인
category 6: 박근혜
category 7: 박원순
category 8: 박지원
category 9: 반기문
category 10: 심상정
category 11: 안철수
category 12: 안희정
category 13: 오세훈
category 14: 유승민
category 15: 유시민
category 16: 이명박
category 17: 이재명
category 18: 홍준표
category 19: 황교안
```

각 뉴스 문서에는 작성 날짜가 기록되어 있기 때문에 시간의 변화에 따른 각 질의어에 대한 토픽의 변화를 추적할 수 있습니다. 이를 위하여 토픽 모델링, 연관어 분석, 문서 요약 등 다양한 방법이 이용될 수 있습니다. 이 repository 는 [politician_news_dataset][politician_news_dataset] 를 이용한 정치인 뉴스를 분석한 예시 코드들이 순차적으로 업데이트 될 계획입니다. 한 접근법에 대한 코드가 정리되면 위의 `latest update` 에 마지막 날짜를 업데이트 하겠습니다.

`tutorials` 폴더의 `config.py` 파일에는 [politician_news_dataset][politician_news_dataset] 데이터셋의 설치 위치가 저장되어 있습니다. 각자의 환경에 맞춰 path 를 변경하신 후 이용하시기 바랍니다.

[politician_news_dataset]: https://github.com/lovit/politician_news_dataset
