
# Classification Model
## Data
- Kor NLI
- klue NLI
## Model
Encoder model
### Backbone
- bert
- roberta
- koelectra

## Experiments

## 실험2
## Data
- https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=59&clCd=ING_TASK&subMenuId=sub01
- 모두의 말뭉치 : 확신성 추론
- task 설명 : premise + hypothesis -> label(float) score 예측
|개수|train|dev|test|
|:------:|:---:|:---:|:---:|
||1448|189|180|
- 여기에서 test는 train에서 sampling해서 구축함

|token 길이|min|median|95%|max|
|:------:|:---:|:---:|:---:|:---:|
||27|92|170|262|

## To do
lora n_labels 1인 경우 해결