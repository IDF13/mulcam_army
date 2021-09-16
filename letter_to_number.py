import pandas as pd

df = pd.read_csv(
    '/Users/sollee/Desktop/sentiment_analysis_TWICE.csv')

df['좋아요'] = df['좋아요'].fillna(0)
value_list = df['좋아요'].values.tolist()
value_list = [str(value) for value in value_list][:1500]  # 데이터 개수 맞춰줄 필요 있음!!!


def letter_to_number():
    result_list = []
    for value in value_list:
        result = 0
        num_map = {'천': 1000, '만': 10000}
        if value.isdigit():
            result = int(value)
        else:
            if len(value) > 1:
                result = float(value[:-1]) * num_map.get(value[-1].upper(), 1)
        result_list.append(result)
    print(result_list)
    return result_list


letter_to_number()
