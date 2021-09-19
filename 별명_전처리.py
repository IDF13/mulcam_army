import pandas as pd
import re

sample = '맏내(맏언니)[6], 미소여왕, 토끼여왕, 고유나연(나나,나영이)[7], 서열 9위, 과즙녀, 나롬[8], 나부기, 나숭, 나봉쓰[9], 백지토[10], 토끼언니[11], 냄새를 맡는 소녀[12], 쪼다연[13], 체리토끼[14], 흰둥이[15], 핫걸[16], 왼쪽 여자[17], MC레일[18], 나봉도치[19][20], 임여워[21], 야망 언니, 나동이[22]'
sample2 = '구름(하구름, 구름이, 구르미)[7], 셍언[8], 떵웅[9], 애기땅콩[10], 8초갑[11], 작소셍[12], 셍남자[13], 위드[14], 셍토끼[15],포메(포메라니안)[16], 작은 거인, 초면갑[17], 하요정[18], 37초남[19], 인간내비[20], 완판남[21], 보부상[22], 하폰지밥[23], 운깅이[24], 행복전도사[25], 하청순, 하애기[26], 촉성운/하셍촉[27], 셍코리타[28], 용맹병아리[29], 막내가장[30], 셍스트[31], 뱉말지남[32], 도토리셍[33]'
regex = "\[*..\]"
re.sub(regex,'',sample2)

# 결과
'''구름(하구름, 구름이, 구르미), 셍언, 떵웅, 애기땅콩,
8초갑, 작소셍, 셍남자, 위드, 셍토끼,포메(포메라니안),
작은 거인, 초면갑, 하요정, 37초남, 인간내비, 완판남,
보부상, 하폰지밥, 운깅이, 행복전도사, 하청순, 하애기,
촉성운/하셍촉, 셍코리타, 용맹병아리, 막내가장, 셍스트, 뱉말지남, 도토리셍'''

df = pd.read_excel('/content/drive/MyDrive/ML Deep/자연어처리/아티스트이름.xlsx')
a = df['별명'].values
len(a)

nick_name = []
for num, nick in enumerate(a):
  regex = "\[*..\]"
  t = re.sub(regex,'',nick)
  df['별명'][num] = t
  

korName = df['korName'].values
engName = df['engName'].values
member = df['member'].values
nick = df['별명'].values

# dictionary1 = dict(zip(korName, member)) # 딕셔너리는 key값이 중복되면 안됨
dictionary = dict(zip(member, nick))


key_name = input('찾으시는 아티스트의 이름을 적으세요')
b =[]
for key, val in dictionary.items():
  if key_name in val:
    b.append(key)

print("찾으시는 단어:",b)

# 결과
'''
찾으시는 아티스트의 이름을 적으세요두부
찾으시는 단어: ['온유', '시은', '윈터', '유용하', '준규', '다현']
'''
