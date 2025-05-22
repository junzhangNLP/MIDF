import eng_to_ipa as p
import re
import pandas as pd
import nltk
from pypinyin import pinyin, Style
import threading
import multiprocessing
from nltk.corpus import stopwords
from tqdm import tqdm
import jieba

def process_data(language, data_name, data_class, sim_threshold=0.7, replace_count_threshold=0.2):
    # 加载数据
    if data_name == "SST-2":
        emoji = pd.read_csv('./dataset/emoji/emoji_all_english.txt', sep=',')
        text = pd.read_csv(f'./dataset/{data_name}/{data_class}.tsv', sep='\t')
        # 预处理emoji数据，将其缓存到字典中
        emoji_dict = {row[1]: row[0] for _, row in emoji.iterrows()}
        emoji_dict_mean = {row[0]: row[1] for _, row in emoji.iterrows()}
        emoji_ipa_dict = {emoji_name: p.convert(emoji_ipa) for emoji_ipa, emoji_name in emoji_dict.items()}

    elif data_name == "IMDB":
        emoji = pd.read_csv('./dataset/emoji/emoji_all_english.txt', sep=',')
        text = pd.read_csv(f'./dataset/{data_name}/{data_class}.tsv', sep='\t')
        # 预处理emoji数据，将其缓存到字典中
        emoji_dict = {row[1]: row[0] for _, row in emoji.iterrows()}
        emoji_dict_mean = {row[0]: row[1] for _, row in emoji.iterrows()}
        emoji_ipa_dict = {emoji_name: p.convert(emoji_ipa) for emoji_ipa, emoji_name in emoji_dict.items()}

    elif data_name == "ChnSentiCorp":
        emoji = pd.read_csv('./dataset/emoji/emoji_all_chinese.txt', sep=',')
        text = pd.read_csv(f'./dataset/{data_name}/{data_class}.tsv', sep='\t')
        emoji_dict = {row[1]: row[0] for _, row in emoji.iterrows()}
        emoji_dict_mean = {row[0]: row[1] for _, row in emoji.iterrows()}
        emoji_ipa_dict = {}  # 创建一个空字典用于存储结果
        for emoji_ipa, emoji_name in emoji_dict.items():
            # 使用 pinyin 函数将 IPA 转换为拼音
            pinyin_result = pinyin(emoji_ipa, style=Style.NORMAL, heteronym=False)
            # 将拼音列表连接成一个字符串，并将结果存储在字典中
            emoji_ipa_dict[emoji_name] = ''.join([p[0] for p in pinyin_result])

    elif data_name == "Weibo_100k":
        emoji = pd.read_csv('./dataset/emoji/emoji_all_chinese.txt', sep=',')
        text = pd.read_csv(f'./dataset/{data_name}/{data_class}.tsv', sep='\t')
        emoji_dict = {row[1]: row[0] for _, row in emoji.iterrows()}
        emoji_dict_mean = {row[0]: row[1] for _, row in emoji.iterrows()}
        emoji_ipa_dict = {}  # 创建一个空字典用于存储结果
        for emoji_ipa, emoji_name in emoji_dict.items():
            # 使用 pinyin 函数将 IPA 转换为拼音
            pinyin_result = pinyin(emoji_ipa, style=Style.NORMAL, heteronym=False)
            # 将拼音列表连接成一个字符串，并将结果存储在字典中
            emoji_ipa_dict[emoji_name] = ''.join([p[0] for p in pinyin_result])
    # 处理数据
    flag = 0
    result_test = []
    total_rows = len(text)
    for index, row in tqdm(text.iterrows(), total=total_rows, desc=f"Processing {data_name} {sim_threshold} {data_class}"):
        if language == "english":
            sentence_without_punctuation = re.sub(r'[^\w\s]', '',row[0])
            sentence_without_punctuation = sentence_without_punctuation.split()
            stop_words = set(stopwords.words('english'))
            words = [word for word in sentence_without_punctuation if word.lower() not in stop_words]
        elif language == "chinese":
            stop_words = set(stopwords.words('chinese'))
            # 去除英文字符
            text_out_eng = re.sub(r'[a-zA-Z]', '', row[0])
            # 去除标点符号
            text_out_bd = re.sub(r'[^\u4e00-\u9fa5]', '', text_out_eng)
            sentence_without_punctuation = re.sub(r'[^\w\s]', '', text_out_bd)
            words = jieba.lcut(sentence_without_punctuation)
            words = [word for word in words if word not in stop_words]
            # # 将句子按句分割
            # words = ''.join([p for p in words])
            # words = [word for word in words]
        old_sentence = row[0]
        word_list = []
        emoji_name_list = []
        emoji_mean_list = []
        replace_count = 0
        replace_count_thr = len(words)*replace_count_threshold
        for word in words:
            if language == "english":
                w_ipa = p.convert(word)
            elif language == "chinese":
                w_ipa = pinyin(word, style=Style.NORMAL, heteronym=False)
                w_ipa = w_ipa[0][0]
                
            for emoji_name, emoji_ipa in emoji_ipa_dict.items():
                lev_distance = nltk.edit_distance(w_ipa, emoji_ipa)
                similarity = 1 - (lev_distance / max(len(w_ipa), len(emoji_ipa)))
                if similarity >= sim_threshold:
                    row[0] = row[0].replace(word, f"[MASK]", 1)
                    # newsentence = row[0].replace(word, f"{emoji_dict_mean[emoji_name]}", 1)
                    word_list.append(word)
                    emoji_name_list.append(emoji_name)
                    emoji_mean_list.append(emoji_dict_mean[emoji_name])
                    replace_count += 1
                    break
            if replace_count >= replace_count_thr:
                break
        if word_list and emoji_name_list and emoji_mean_list:
            result_test.append((old_sentence, row[0], word_list, emoji_name_list, emoji_mean_list, row[1]))
    # 将结果保存到DataFrame中
    result_df = pd.DataFrame(result_test, columns=["old_text", 'text', 'word', 'emoji', "emoji_mean", "lables"])

    # 输出结果
    result_df.to_csv(f'./dataset/Dataset_ipa/{language}/{data_name}_{data_class}_{sim_threshold}_{replace_count_threshold}.txt',
                     sep='|', encoding='UTF-8', index=False)


if __name__ == "__main__":
    num_class = ["test", "dev", "train"]
    # num_class = ["train"]
    sim = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # sim = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    replace_th = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    for j in num_class:
        processes = []
        for i in sim:
            # process = threading.Thread(target=process_data, args=("english", 'SST-2', j, i, 0.4))
            # process = multiprocessing.Process(target=process_data, args=("chinese", 'ChnSentiCorp', j, i, 0.4))
            # process = multiprocessing.Process(target=process_data, args=("english", 'IMDB', j, i, 0.4))
            process = multiprocessing.Process(target=process_data, args=("chinese", 'Weibo_100k', j, i, 0.4))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    for j in num_class:
        processes = []
        for i in replace_th:
            # process = threading.Thread(target=process_data, args=("english", 'SST-2', j, 0.7, i))
            # process = multiprocessing.Process(target=process_data, args=("chinese", 'ChnSentiCorp', j, 0.7, i))
            # process = multiprocessing.Process(target=process_data, args=("english", 'IMDB', j,  0.7, i))
            process = multiprocessing.Process(target=process_data, args=("chinese", 'Weibo_100k', j,  0.7, i))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
    # process_data("chinese", 'ChnSentiCorp', "train", 0.4, 0.2)
    # process_data("english", 'SST-2', "dev", 0.4, 0.2)
    # process_data("chinese", 'Weibo_100k', "train", 0.4, 0.2)