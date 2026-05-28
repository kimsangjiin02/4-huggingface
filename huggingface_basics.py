#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers datasets huggingface_hub')


# # 1. 허깅페이스 개요

# ### BERT와 GPT-2 모델을 활용할 때 허깅페이스 트랜스포머 코드 비교

# In[ ]:


from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel

text = "What is Huggingface Transformers?"
# BERT 모델 활용
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded_input = bert_tokenizer(text, return_tensors='pt')
bert_output = bert_model(**encoded_input)
# GPT-2 모델 활용
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')
encoded_input = gpt_tokenizer(text, return_tensors='pt')
gpt_output = gpt_model(**encoded_input)


# In[ ]:


bert_output #출력값: 입력 텍스트의 각 토큰과 전체 시퀀스를 나타내는 고차원의 벡터(은닉 상태)
gpt_output #출력값: 로짓(logits) 벡터. 각 단어(토큰)가 다음에 나올 확률을 나타내는 값
#BERT: 분류 모델로, 텍스트화 불가, GPT: 다음 단어 예측하는 모델로, 텍스트 출력하려면 별도 작업 필요


# # 2. 트랜스포머 모델 활용하기

# ### 모델 불러오기

# ### 모델 아이디로 바디만 불러오기

# In[ ]:


from transformers import AutoModel
base_model_id = 'bert-base-uncased'
base_model =AutoModel.from_pretrained(base_model_id)


# In[ ]:


#model metadata
base_model_config = base_model.config
config_dict = base_model_config.__dict__
config_dict


# ### 분류 헤드가 포함된 모델 불러오기

# In[ ]:


from transformers import AutoModelForSequenceClassification
classification_model_id = 'SamLowe/roberta-base-go_emotions'
classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_id)


# In[ ]:


#model metadata
classification_model_config = classification_model.config
config_dict = classification_model_config.__dict__
config_dict
#id2label확인


# ### 분류 헤드가 랜덤으로 초기화된 모델 불러오기

# In[ ]:


from transformers import AutoModelForSequenceClassification
classification_model_id = 'SamLowe/roberta-base-go_emotions'
classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_id)


# In[ ]:


# model metadata
classification_model_config = classification_model.config
config_dict = classification_model_config.__dict__
config_dict
# id2label 확인
classification_model.config.id2label


# ## 토크나이저 활용하기

# ### 토크나이저 사용하기

# In[ ]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenizer("I love banana")
print(encoded)
# {
#   'input_ids':      [101, 1045, 2293, 15212, 102],
#   'token_type_ids': [0, 0, 0, 0, 0],
#   'attention_mask': [1, 1, 1, 1, 1]
# }


# In[ ]:


# 토큰화 결과 확인
print(tokenizer.convert_ids_to_tokens([101, 1045, 2293, 15212, 102]))
# ['[CLS]', 'i', 'love', 'banana', '[SEP]']

# 다시 텍스트로
print(tokenizer.decode([101, 1045, 2293, 15212, 102], skip_special_tokens=True))
# 'i love banana'


# # 3. 데이터셋 활용하기

# ## 데이터셋 다운로드

# ### 허깅페이스 허브에서 데이터셋 다운로드

# In[ ]:


from datasets import load_dataset
#MRC(Machine Reading Comprehension) 데이터셋: 기계의 독해 능력 평가하는 데이터셋
klue_mrc_dataset = load_dataset('klue/klue', 'mrc')
klue_mrc_dataset_only_train = load_dataset('klue/klue', 'mrc', split='train')


# In[ ]:


klue_mrc_dataset


# In[ ]:


klue_mrc_dataset_only_train


# In[ ]:


klue_mrc_dataset['train'][0]


# ### 로컬의 데이터 활용하기

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')  # ← 드라이브 마운트로 교체
dataset_json = load_dataset("json", data_files="/content/drive/MyDrive/git/4-huggingface/review_신라스테이 해운대.json")

dataset_json_train = load_dataset("json", data_files="/content/drive/MyDrive/git/4-huggingface/review_신라스테이 해운대.json", split='train')

dataset_json_test = dataset_json["train"].train_test_split(test_size=0.2, seed=42)["test"]


# 
# ### 데이터셋 제작

# In[ ]:


# 파이썬 딕셔너리 활용
from datasets import Dataset
my_dict = {"a": [1, 2, 3]}
dataset = Dataset.from_dict(my_dict)
print(dataset[0])

# 판다스 데이터프레임 활용
from datasets import Dataset
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3]})
dataset = Dataset.from_pandas(df)
print(dataset[0])


# ## 데이터셋 가공하기

# ### 실습에 사용하지 않는 불필요한 컬럼 제거

# In[ ]:


from datasets import load_dataset
klue_tc_train = load_dataset("klue/klue", "ynat", split="train")
klue_tc_eval = load_dataset("klue/klue", "ynat", split="validation")
print(klue_tc_train)
print(klue_tc_eval)


# In[ ]:


klue_tc_train_removed = klue_tc_train.remove_columns(['guid', 'url', 'date'])
klue_tc_eval_removed  = klue_tc_eval.remove_columns(['guid', 'url', 'date'])

print(klue_tc_train_removed)
print(klue_tc_train_removed[0])


# ### 카테고리를 문자로 표기한 label_str 컬럼 추가

# In[ ]:


print(klue_tc_train_removed.features['title'])
print(klue_tc_train_removed.features['label'])


# In[ ]:


klue_tc_label = klue_tc_train_removed.features['label']

def make_str_label(batch):
    batch['label_str'] = klue_tc_label.int2str(batch['label'])
    return batch

klue_tc_train_removed = klue_tc_train_removed.map(make_str_label, batched=True, batch_size=1000)
klue_tc_eval_removed  = klue_tc_eval_removed.map(make_str_label, batched=True, batch_size=1000)

klue_tc_train_removed[10]


# ### 학습/검증/테스트 데이터셋 분할

# In[ ]:


train_dataset = klue_tc_train_removed.train_test_split(test_size=10000, shuffle=True, seed=42)['test']
dataset = klue_tc_eval_removed.train_test_split(test_size=1000, shuffle=True, seed=42)
test_dataset = dataset['test']
valid_dataset = dataset['train'].train_test_split(test_size=1000, shuffle=True, seed=42)['test']


# # 4. 모델을 이용하여 추론하기

# ### 학습한 모델을 불러와 pipeline을 활용해 텍스트 분류하기

# In[ ]:


from transformers import pipeline

model_id = "hykiim/roberta-base-klue-ynat-classification"

model_pipeline = pipeline("text-classification", model=model_id)

model_pipeline(test_dataset["title"][:5])


# ### 커스텀 파이프라인 구현

# In[ ]:


import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CustomPipeline:
    def __init__(self, model_id):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)#목적에 맞는 모델 헤드 불러옴
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)#모델과 동일한 토크나이저 불러옴
        self.model.eval() #모델 평가 모드(아직은 몰라도 됨.)

    def __call__(self, texts):
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)# 입력 텍스트를 토크나이저를 사용하여 토큰화
        # return_tensors="pt": PyTorch 텐서로 반환, padding=True: 패딩 추가, truncation=True: 잘라내기

        with torch.no_grad():#기울기 계산 비활성화(옵티마이저 계산 생략)-> 추론 과정에서는 옵티마이징 할 필요 없음.
            outputs = self.model(**tokenized)#**연산자는 딕셔너리를 키워드 인자로 unpacking하여 모델에 입력. 즉 토큰 아이디
            logits = outputs.logits# 모델 출력에서 logits 값을 추출. (분류되지 않은 예측값)

        #추론이 정확할 확률 구하기
        probabilities = softmax(logits, dim=-1)
        scores, labels = torch.max(probabilities, dim=-1)
        labels_str = [self.model.config.id2label[label_idx] for label_idx in labels.tolist()]

        return [{"label": label, "score": score.item()} for label, score in zip(labels_str, scores)]

custom_pipeline = CustomPipeline(model_id)
print(custom_pipeline(test_dataset['title'][:5]))
print(test_dataset['label_str'][:5])


# In[ ]:


from transformers import pipeline
import torch

# 텍스트 생성 파이프라인 로드
# model 인자에 'gpt2'를 지정합니다. 다른 gpt2 변형(gpt2-medium, gpt2-large 등)도 사용 가능합니다.
# device=0 은 GPU 사용을 의미합니다. GPU가 없거나 CPU를 사용하려면 device=-1 또는 이 인자를 생략합니다.
generator = pipeline('text-generation', model='gpt2', device=0) # GPU 사용 시
#generator = pipeline('text-generation', model='gpt2') # CPU 사용 시 (기본값)

# 텍스트 생성을 시작할 프롬프트(prompt) 또는 시작 문맥(context)
prompt = "Once upon a time, in a land far, far away,"

# 파이프라인을 사용하여 텍스트 생성 실행
# max_length: 생성될 텍스트의 최대 길이 (프롬프트 포함)
# num_return_sequences: 몇 개의 다른 생성 결과를 반환할지
# 결과는 리스트 형태로 반환되며, 각 요소는 딕셔너리 형태입니다.
results = generator(prompt, max_length=50, num_return_sequences=1)

# 생성된 텍스트 출력
print(f"\n프롬프트: {prompt}")
print("--- 생성된 텍스트 ---")
for result in results:
    print(result['generated_text'])

print("-" * 20)

# 여러 개의 결과 생성 및 다양한 옵션 사용 예시
prompt_2 = "The future of Artificial Intelligence is"
results_options = generator(
    prompt_2,
    max_length=100,           # 더 긴 텍스트 생성
    num_return_sequences=3,  # 3개의 다른 결과 요청
    do_sample=True,          # 샘플링 사용 (더 창의적인 결과)
    temperature=0.7,         # 낮은 온도는 더 예측 가능한 텍스트 생성
    top_k=50,                # top-k 샘플링: 확률 높은 상위 50개 단어 중에서 선택
    no_repeat_ngram_size=2   # 2-gram 반복 방지
)

print(f"\n프롬프트: {prompt_2}")
print("--- 다양한 옵션으로 생성된 텍스트 (3개) ---")
for i, result in enumerate(results_options):
    print(f"결과 {i+1}:")
    print(result['generated_text'])
    print("-" * 10)


# In[ ]:


import torch

gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')
encoded_input = gpt_tokenizer(text, return_tensors='pt')
gpt_output = gpt_model(**encoded_input)


# GPT-2는 기본 pad 토큰이 없으므로, eos 토큰을 pad 토큰으로 설정 (일반적인 관행)
if gpt_tokenizer.pad_token is None:
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

# 2. 입력 텍스트 준비
text = "What is Huggingface Transformer?"
encoded_input = gpt_tokenizer(text, return_tensors='pt')
input_ids = encoded_input['input_ids']

# 3. 텍스트 생성 (model.generate 사용)
# max_length: 생성될 텍스트의 최대 길이 (입력 포함)
# num_return_sequences: 생성할 시퀀스 수
# pad_token_id: 패딩에 사용될 토큰 ID 설정
output_sequences = gpt_model.generate(
    input_ids=input_ids,
    max_length=50,  # 예시: 최대 50 토큰까지 생성
    num_return_sequences=1,
    pad_token_id=gpt_tokenizer.pad_token_id,
    # 더 다양한 결과를 원하면 다음 파라미터 추가 가능:
    do_sample=True,      # 샘플링 사용 여부
    top_k=50,            # 상위 K개 토큰 중에서만 샘플링
    top_p=0.95,          # 누적 확률 P 이상인 토큰 중에서만 샘플링 (nucleus sampling)
    temperature=0.7,     # 확률 분포를 조절 (낮을수록 결정적, 높을수록 무작위적)
)

# 4. 생성된 토큰 ID 시퀀스를 텍스트로 디코딩
# output_sequences[0]은 생성된 첫 번째 시퀀스를 의미
# skip_special_tokens=True 옵션은 <|endoftext|> 같은 특수 토큰을 결과에서 제외
generated_text = gpt_tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# 5. 결과 출력
print("Input Text:", text)
print("Generated Text:", generated_text)


# In[ ]:


# @title 1. 환경 설정 및 라이브러리 설치 (Cell 1)
"""
이 셀에서는 실습에 필요한 라이브러리들을 설치하고 임포트합니다.
- transformers: Hugging Face 모델 및 파이프라인 사용을 위한 라이브러리
- datasets: KLUE 데이터셋 등 Hugging Face Hub의 데이터셋 로드를 위한 라이브러리
- sentencepiece: KoBART 등 일부 모델에서 사용하는 토크나이저 라이브러리
- accelerate: 모델 로딩 및 분산 처리를 도와주는 라이브러리 (특히 NLLB 모델에 유용)
- torch: PyTorch 라이브러리 (기본 백엔드, Colab 기본 설치)
"""

# 안정 버전으로 설치 (v5 호환성 문제 회피)

get_ipython().system('pip install -q "transformers<5.0" "tokenizers<0.21" datasets sentencepiece accelerate')

import datasets
from datasets import load_dataset, DatasetDict
from transformers import pipeline
import torch
import pandas as pd

# GPU 사용 가능 여부 확인 및 설정
device = 0 if torch.cuda.is_available() else -1
print(f"사용 가능한 디바이스: {'GPU' if device == 0 else 'CPU'}")


# In[ ]:


# @title 2. 데이터셋 로드 및 준비 (Cell 2)
"""
이 셀에서는 KLUE 데이터셋의 MRC 부분을 로드합니다.
전체 데이터셋은 클 수 있으므로, 실습을 위해 일부 데이터만 선택하여 사용합니다.
'context' 컬럼이 우리가 요약할 원본 기사 내용입니다.
"""
full_dataset = load_dataset("klue/klue", "mrc", split="train")

# 실습을 위해 데이터 일부만 선택 (예: 앞 10개)
num_samples_to_use = 10
klue_mrc_subset = full_dataset.select(range(num_samples_to_use))

print("로드된 데이터셋 정보:")
print(klue_mrc_subset)

print("\n첫 번째 데이터 예시 (context 확인):")
print(klue_mrc_subset[0]['context'])

# 데이터셋 확인을 위해 Pandas DataFrame으로 변환 (선택 사항)
df_check = pd.DataFrame(klue_mrc_subset)
print("\n데이터셋 일부 미리보기 (DataFrame):")
display(df_check.head(3)) # Colab 환경에서는 display()가 표 형태로 보여줍니다.


# In[ ]:


get_ipython().system('pip install transformers==4.40.0 huggingface_hub==0.23.0 -q')


# In[ ]:


# @title 3. 요약 모델 파이프라인 로드 (Cell 3)
"""
이 셀에서는 기사 요약을 위한 KoBART 기반 모델 파이프라인을 로드합니다.
모델: gogamza/kobart-summarization
파이프라인 타입: summarization
"""
from transformers import pipeline

summarizer = pipeline(
    task="summarization",
    model="gogamza/kobart-summarization",
    device=device
)
print("요약 파이프라인 로드 완료.")


# In[ ]:


# @title 4. 기사 내용 요약 및 데이터셋에 추가 (Cell 4)
"""
이 셀에서는 로드된 요약 파이프라인을 사용하여 데이터셋의 'context' 내용을 요약합니다.
map 함수를 사용하여 데이터셋의 각 샘플에 요약 함수를 적용하고,
결과를 'summary'라는 새로운 컬럼에 저장합니다.
"""
# 요약을 수행하는 함수 정의
def summarize_context(example):
  """데이터셋의 'context'를 받아 요약 결과를 반환하는 함수"""
  summary_result = summarizer(
        example['context'],
        max_length=150,
        min_length=30,
        do_sample=False  # deterministic 출력을 원하면 False, 다양성을 원하면 True
    )
    # 결과에서 요약 텍스트 추출
  example['summary'] = summary_result[0]['summary_text']
  return example

print("요약 작업을 시작합니다... (데이터 양에 따라 시간이 소요될 수 있습니다)")
summarized_dataset = klue_mrc_subset.map(summarize_context)
print("요약 작업 완료.")

print("\n요약이 추가된 데이터셋 정보:")
print(summarized_dataset)

print("\n첫 번째 데이터의 원문(context)과 요약(summary):")
print("--- 원문 (Context) ---")
print(summarized_dataset[0]['context'])
print("\n--- 요약 (Summary) ---")
print(summarized_dataset[0]['summary'])

# 데이터셋 확인 (Pandas)
df_check_summary = pd.DataFrame(summarized_dataset)
print("\n요약 추가 후 데이터셋 미리보기 (DataFrame):")
display(df_check_summary.head(3))


# In[ ]:


# @title 5. 번역 모델 파이프라인 로드 (Cell 5)
"""
이 셀에서는 한국어 요약본을 영어로 번역하기 위한 NLLB 모델 파이프라인을 로드합니다.
모델: facebook/nllb-200-distilled-600M
파이프라인 타입: translation
NLLB 모델은 다양한 언어를 지원하며, 언어 코드를 지정해야 합니다.
한국어: kor_Hang, 영어: eng_Latn
"""
translator = pipeline(
    task="translation",
    model="facebook/nllb-200-distilled-600M",
    device=device
)

print("번역 파이프라인 로드 완료.")

# 번역 테스트 (선택 사항)
test_translation = translator(
    "안녕하세요?",
    src_lang="kor_Hang",
    tgt_lang="eng_Latn"
)


# In[ ]:


# @title 6. 기사 요약본 영어로 번역 및 데이터셋에 추가 (Cell 6)
"""
이 셀에서는 생성된 'summary' 컬럼의 한국어 텍스트를 영어로 번역합니다.
map 함수를 사용하여 번역 함수를 적용하고,
결과를 'english_summary'라는 새로운 컬럼에 저장합니다.
"""
# 번역을 수행하는 함수 정의
def translate_summary_to_english(example):
  """데이터셋의 'summary'를 받아 영어 번역 결과를 반환하는 함수"""
  translation_result = translator(
        example['summary'],
        src_lang="kor_Hang",
        tgt_lang="eng_Latn",
        max_length=150,
        min_length=30,
        do_sample=False  # deterministic 출력을 원하면 False, 다양성을 원하면 True
    )
    # 결과에서 번역 텍스트 추출
  example['english_summary'] = translation_result[0]['translation_text']
  return example

print("번역 작업을 시작합니다... (모델 크기와 데이터 양에 따라 시간이 많이 소요될 수 있습니다)")
translated_dataset = summarized_dataset.map(translate_summary_to_english)
print("번역 작업 완료.")

print("\n번역이 추가된 데이터셋 정보:")
print(translated_dataset)

print("\n첫 번째 데이터의 한국어 요약(summary)과 영어 번역(english_summary):")
print("--- 한국어 요약 (Summary) ---")
print(translated_dataset[0]['summary'])
print("\n--- 영어 번역 (English Summary) ---")
print(translated_dataset[0]['english_summary'])

# 데이터셋 확인 (Pandas)
df_check_translation = pd.DataFrame(translated_dataset)
print("\n번역 추가 후 데이터셋 미리보기 (DataFrame):")
display(df_check_translation.head(3))


# In[ ]:


# @title 7. 감정 분석 모델 파이프라인 로드 (Cell 7)
"""
이 셀에서는 영어 텍스트의 감정을 분석하기 위한 모델 파이프라인을 로드합니다.
모델: SamLowe/roberta-base-go_emotions
파이프라인 타입: text-classification
이 모델은 다중 레이블 감정(분노, 기쁨, 슬픔 등)을 예측할 수 있습니다.
"""
emotion_classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=1,
    device=device
)
print("감정 분석 파이프라인 로드 완료.")

# 감정 분석 테스트 (선택 사항)
test_emotion = emotion_classifier("I am very happy today!")
print(f"감정 분석 테스트: {test_emotion[0][0]}")


# In[ ]:


# @title 8. 영어 번역본 감정 분석 및 데이터셋에 추가 (Cell 8)
"""
이 셀에서는 번역된 'english_summary' 컬럼의 텍스트에 대해 감정 분석을 수행합니다.
map 함수를 사용하여 감정 분석 함수를 적용하고,
가장 확률이 높은 감정 레이블을 'emotion'이라는 새로운 컬럼에 저장합니다.
"""
# 감정 분석을 수행하는 함수 정의
def analyze_emotion(example):
  """데이터셋의 'english_summary'를 받아 감정 분석 결과를 반환하는 함수"""
  emotion_result = emotion_classifier(example['english_summary'])
    # 결과는 [[{'label': '...', 'score': ...}]] 형태 (top_k=1이므로 리스트 길이 1)
  example['emotion'] = emotion_result[0][0]['label']
  return example

print("감정 분석 작업을 시작합니다...")
final_dataset = translated_dataset.map(analyze_emotion)
print("감정 분석 작업 완료.")

print("\n최종 데이터셋 정보:")
print(final_dataset)

print("\n첫 번째 데이터의 영어 번역(english_summary)과 감정(emotion):")
print("--- 영어 번역 (English Summary) ---")
print(final_dataset[0]['english_summary'])
print("\n--- 분석된 감정 (Emotion) ---")
print(final_dataset[0]['emotion'])

# 최종 데이터셋 확인 (Pandas)
df_final = pd.DataFrame(final_dataset)
print("\n최종 데이터셋 미리보기 (DataFrame):")
display(df_final) # 전체 선택된 샘플 표시


# In[ ]:


# @title 9. 결과 정리 및 마무리 (Cell 9)
"""
모든 단계를 거쳐 생성된 최종 데이터셋(final_dataset)에는
원본 KLUE-MRC 데이터에 'summary', 'english_summary', 'emotion' 컬럼이 추가되었습니다.
이 데이터를 CSV 파일로 저장하거나 추가 분석에 활용할 수 있습니다.
"""
print("모든 작업이 완료되었습니다.")
print("최종 데이터셋 컬럼:", final_dataset.column_names)

# 최종 결과 확인 (첫 5개 샘플)
for i in range(min(5, len(final_dataset))):
  print(f"\n--- 샘플 {i+1} ---")
  print(f"원문 일부: {final_dataset[i]['context'][:100]}...")
  print(f"요약: {final_dataset[i]['summary']}")
  print(f"영어 번역: {final_dataset[i]['english_summary']}")
  print(f"감정 분석: {final_dataset[i]['emotion']}")


# In[ ]:


print(final_dataset.column_names)


# In[ ]:


과제_ 기사 번역


# In[ ]:


import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from datasets import load_dataset
from transformers import pipeline
from IPython.display import display

# 한글 폰트 설정 (Colab 환경)
get_ipython().system('apt-get -qq install fonts-nanum')
matplotlib.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False

# GPU/CPU 설정
device = 0 if torch.cuda.is_available() else -1
device_name = 'GPU ✅' if device == 0 else 'CPU ⚠️ (GPU 권장)'
print(f"사용 디바이스: {device_name}")
print(f"PyTorch 버전: {torch.__version__}")


# In[ ]:


# KLUE-MRC 데이터셋 로드
print("데이터셋을 로드하는 중...")
full_dataset = load_dataset("klue/klue", "mrc", split="train")

# 실습용 서브셋 (10개)
NUM_SAMPLES = 10
subset = full_dataset.select(range(NUM_SAMPLES))

print(f"로드 완료: {NUM_SAMPLES}개 샘플 선택")
print(f"\n데이터셋 구조:\n{subset}")
print(f"\n컬럼 목록: {subset.column_names}")


# In[ ]:


# 첫 번째 샘플 상세 확인
sample = subset[0]
print("=" * 60)
print("첫 번째 샘플 미리보기")
print("=" * 60)
print(f"[제목] {sample['title']}")
print(f"\n[질문] {sample['question']}")
print(f"\n[본문 (앞 300자)]\n{sample['context'][:300]}...")

# Pandas로 전체 미리보기
df_raw = pd.DataFrame(subset).rename(columns={"context": "본문", "title": "제목"})
display(df_raw[["제목", "본문"]].applymap(lambda x: str(x)[:80] + "..." if len(str(x)) > 80 else x))


# In[ ]:


get_ipython().system('pip install -q "transformers>=4.41.0" "tokenizers<0.21"')


# In[ ]:


# 요약 파이프라인 로드
from transformers import pipeline
print("요약 모델 로드 중... (첫 실행 시 시간이 걸릴 수 있습니다)")
summarizer = pipeline(
    task="summarization",
    model="gogamza/kobart-summarization",
    device=device
)
print("✅ 요약 파이프라인 로드 완료")

# 빠른 테스트
test_text = subset[0]['context']
test_result = summarizer(test_text, max_length=150, min_length=30, do_sample=False)
print(f"\n[테스트 요약 결과]\n{test_result[0]['summary_text']}")


# In[ ]:


# 요약 함수 정의
def summarize_context(example):
    """context 컬럼을 요약하여 summary 컬럼으로 반환"""
    result = summarizer(
        example['context'],
        max_length=150,
        min_length=30,
        do_sample=False
    )
    example['summary'] = result[0]['summary_text']
    return example

# 전체 서브셋에 적용
print("요약 작업 시작...")
summarized_dataset = subset.map(summarize_context)
print("요약 완료!")

# 결과 확인
print("\n" + "=" * 60)
for i in range(3):
    print(f"\n[샘플 {i+1}] 원문 (앞 150자):")
    print(f"  {summarized_dataset[i]['context'][:150]}...")
    print(f"  → 요약: {summarized_dataset[i]['summary']}")


# In[ ]:


from transformers import pipeline
# 번역 파이프라인 로드
print("번역 모델 로드 중... (모델이 커서 시간이 걸립니다)")
translator = pipeline(
    task="translation",
    model="facebook/nllb-200-distilled-600M",
    device=device
)
print("번역 파이프라인 로드 완료")

# 테스트
test_trans = translator("안녕하세요, 오늘 날씨가 좋습니다.", src_lang="kor_Hang", tgt_lang="eng_Latn")
print(f"\n[번역 테스트] 안녕하세요 → '{test_trans[0]['translation_text']}'")


# In[ ]:


# 번역 함수 정의
def translate_summary(example):
    """summary(한국어)를 english_summary(영어)로 번역"""
    result = translator(
        example['summary'],
        src_lang="kor_Hang",
        tgt_lang="eng_Latn",
        max_length=200
    )
    example['english_summary'] = result[0]['translation_text']
    return example

# 전체 서브셋에 적용
print("번역 작업 시작...")
translated_dataset = summarized_dataset.map(translate_summary)
print("번역 완료!")

# 결과 확인
print("\n" + "=" * 60)
for i in range(3):
    print(f"\n[샘플 {i+1}]")
    print(f"  한국어 요약: {translated_dataset[i]['summary']}")
    print(f"  영어 번역:  {translated_dataset[i]['english_summary']}")


# In[ ]:


# 감정 분석 파이프라인 로드
print("감정 분석 모델 로드 중...")
emotion_classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=1,
    device=device
)
print("감정 분석 파이프라인 로드 완료")

# 테스트
tests = ["I feel so happy today!", "This is really disappointing.", "The results were surprising."]
for t in tests:
    r = emotion_classifier(t)
    print(f"  '{t}'\n   → {r[0][0]['label']} (score: {r[0][0]['score']:.3f})")


# In[ ]:


# 감정 분석 함수 정의
def analyze_emotion(example):
    """english_summary에 감정 분석을 수행하여 emotion 컬럼으로 반환"""
    result = emotion_classifier(example['english_summary'])
    top = result[0][0]
    example['emotion'] = top['label']
    example['emotion_score'] = round(top['score'], 4)
    return example

# 전체 서브셋에 적용
print("감정 분석 작업 시작...")
final_dataset = translated_dataset.map(analyze_emotion)
print("감정 분석 완료!")

# 결과 확인
print("\n" + "=" * 60)
for i in range(5):
    print(f"\n[샘플 {i+1}]")
    print(f"  영어 번역:  {final_dataset[i]['english_summary'][:100]}...")
    print(f"  감정: {final_dataset[i]['emotion']}  (확신도: {final_dataset[i]['emotion_score']:.3f})")


# In[ ]:


# 최종 데이터셋을 DataFrame으로 변환
df = pd.DataFrame(final_dataset)[['title', 'context', 'summary', 'english_summary', 'emotion', 'emotion_score']]
df.columns = ['제목', '원문', '한국어_요약', '영어_번역', '감정', '감정_확신도']

print("최종 데이터셋 컬럼:", df.columns.tolist())
print(f"총 샘플 수: {len(df)}\n")

# 전체 결과 출력
pd.set_option('display.max_colwidth', 60)
display(df[['제목', '한국어_요약', '영어_번역', '감정', '감정_확신도']])


# In[ ]:


# ── 시각화 1: 감정 분포 파이 차트 ──
emotion_counts = df['감정'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("KLUE-MRC 뉴스 기사 감정 분석 결과", fontsize=15, fontweight='bold', y=1.02)

# 파이 차트
colors = plt.cm.Set3.colors[:len(emotion_counts)]
axes[0].pie(
    emotion_counts.values,
    labels=emotion_counts.index,
    autopct='%1.0f%%',
    colors=colors,
    startangle=140,
    textprops={'fontsize': 11}
)
axes[0].set_title("감정 레이블 비율", fontsize=12, pad=10)

# 바 차트 (확신도)
bar_colors = [f'C{i}' for i in range(len(df))]
bars = axes[1].barh(
    [f"샘플 {i+1}" for i in range(len(df))],
    df['감정_확신도'],
    color=bar_colors
)
axes[1].set_xlim(0, 1)
axes[1].set_xlabel("감정 확신도 (Score)", fontsize=11)
axes[1].set_title("샘플별 감정 예측 확신도", fontsize=12)

# 각 바에 레이블
for bar, label in zip(bars, df['감정']):
    axes[1].text(
        bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
        label, va='center', fontsize=9
    )

plt.tight_layout()
plt.savefig("emotion_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("시각화 저장 완료 (emotion_analysis.png)")


# In[ ]:


# ── 상세 출력: 파이프라인 전 과정 요약 ──
print("=" * 70)
print("🏁 최종 파이프라인 결과 - 전체 샘플 상세 보기")
print("=" * 70)

for i, row in df.iterrows():
    print(f"\n{'─'*60}")
    print(f"📰 [샘플 {i+1}] {row['제목']}")
    print(f"\n  [원문 (앞 120자)]")
    print(f"  {row['원문'][:120]}...")
    print(f"\n  [한국어 요약]")
    print(f"  {row['한국어_요약']}")
    print(f"\n  [영어 번역]")
    print(f"  {row['영어_번역']}")
    print(f"\n  [감정 분석] → {row['감정'].upper()}  (확신도: {row['감정_확신도']:.3f})")

print(f"\n{'=' * 70}")
print("모든 파이프라인 단계 완료!")
print(f"  1단계 요약    : gogamza/kobart-summarization")
print(f"  2단계 번역    : facebook/nllb-200-distilled-600M")
print(f"  3단계 감정분석: SamLowe/roberta-base-go_emotions")
print(f"  처리 샘플 수  : {len(df)}개")


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('ls /content/drive/MyDrive/git')


# In[ ]:


get_ipython().system('jupyter nbconvert --to script *.ipynb')


# In[ ]:


# 현재 변경된 내용을 commit 적용항목에 포함
get_ipython().system('git add .')
# add 된 항목, commit 처리
get_ipython().system('git commit -m "initial commit!"')


# In[ ]:


# commit 상태 확인
get_ipython().system('git status')


# In[ ]:


get_ipython().system('git push origin 김상진_week10')

# 강제 push (local을 강제로 원격으로 적용)#!git push -f origin main

