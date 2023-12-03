#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model.ai_loader import client
from langchain.prompts import ChatPromptTemplate
from utils import read_jsonl
from config import project_config
import pandas as pd
import os
import json


def extract_company_names():
    df = pd.DataFrame(columns=['filename', 'company'])
    i = 1
    filepaths = os.listdir(project_config.text_files_path)
    filepaths.sort()
    for filename in filepaths:
        if filename.endswith(".txt"):
            file_path = os.path.join(project_config.text_files_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                template = ChatPromptTemplate.from_template(
                    "你是一个能精准提取信息的AI。"
                    "我会给你一篇招股说明书部分内容，请指出它的主体是哪家公司（一般发行人或者发行人简要情况后面的公司名称也是主体）。"
                    "如果找到公司，则输出公司名称, 输出格式例如 主体是：xxx\n\n"
                    "{t}\n\n"
                )
                max_token = 3000
                idx = content.find("发行人：")
                if idx > 0 :
                    prompt = template.format_messages(t=content[idx:idx+max_token])
                else:
                    idx = content.rfind("发行人简要情况")
                    if idx > 0 and content[idx:idx+100].find(".......") < 0 : # 排除章节
                        prompt = template.format_messages(t=content[idx:idx+max_token])
                    else:
                        prompt = template.format_messages(t=content[:max_token])
                
                resp = client.chat.completions.create(model=project_config.model_name,
                                                      messages=[{"role": "user", "content": prompt[0].content}],
                                                      temperature=project_config.temperature, top_p=project_config.top_p)
                result = resp.choices[0].message.content
                # temp 临时人工处理这些公司：
                # elif filename == 'd336d607e1d431cbfe1f313e2234a13fcf49a16e.txt':
                #     result = '湖南国科微电子股份有限公司'
                # elif filename == 'a6f8156c08a1096c46470a1c5e1229daaaedf06e.txt':
                #     result = '成都华气厚普机电设备股份有限公司'
                # elif filename == '54d148902b889679830174597830f0d0f22c1073.txt':
                #     result = '上海派能能源科技股份有限公司'
                
                if filename == '91b4426b075560a1a45247f9cfa9fa73d56c945c.txt':
                    result =  "广州中海达卫星导航技术股份有限公司"
                    alias = result
                elif filename == 'afa8c5a4a91c3ecf7bd38a1c1f09b8a68e472909.txt':
                    result = '海看网络科技（山东）股份有限公司' 
                    alias = '山东海看网络科技有限公司'
                elif filename == '28560e1383141e35127388a3f0ca0e7b24919c17.txt':
                    result = '江苏旷达汽车织物集团股份有限公司'
                    alias = '旷达汽车织物集团'
                    # 中国铁路通信信号股份有限公司
                elif filename == '96b461d6c6670928f7dc36f0c947e0c18340d5e2.txt':
                    result = '湖南南岭民用爆破器材股份有限公司'
                    alias = '南岭化工厂'
                elif result.startswith("主体是："):
                    result = result[4:]
                    alias = result
                elif result.startswith("主体是"):
                    result = result[3:]
                    alias = result
                
                # 空格处理
                result = result.replace(" ", "")
                df.at[i, 'filename'] = filename
                df.at[i, 'company'] = result
                df.at[i, 'alias'] = alias
                i += 1
                print("i=" + str(i - 1) + ",filename=" + filename + ",rst=" + result)
    df.to_csv(project_config.save_path)


def split_questions_by_type():
    questions = read_jsonl(project_config.questions_path)
    df = pd.read_csv(project_config.company_file_path)
    company_list = df['company']
    company_alias_list = df['alias']

    df_text = pd.DataFrame(columns=['id', 'question', 'company'])
    df_data = pd.DataFrame(columns=['id', 'question'])
    i_text, i_data = 0, 0

    for question in questions:
        flag = True
        q = question['question']
        for idx, company in enumerate(company_list):
            alias = company_alias_list[idx]
            if company in q or alias in q:
                df_text.at[i_text, 'id'] = question['id']
                df_text.at[i_text, 'question'] = question['question']
                df_text.at[i_text, 'company'] = company
                flag = False
                i_text += 1
                break
        if flag:
            df_data.at[i_data, 'id'] = question['id']
            df_data.at[i_data, 'question'] = question['question']
            i_data += 1

    df_text.to_csv(project_config.text_questions_path)
    df_data.to_csv(project_config.data_questions_path)


def integrate_outputs():
    # 读取文本理解和数据查询的答案文件
    data = []
    i = 0
    i1 = 0
    i2 = 0
    df1 = pd.read_csv(project_config.text_answer_path,index_col=0)[['id', 'question', 'answer']]
    df2 = pd.read_csv(project_config.data_answer_path,index_col=0)[['id', 'question', 'formatted_answer']]
    df2.columns = ['id', 'question', 'answer']
    #print(df1.columns)
    #print(df2.columns)

    while i < 1000:
        try:
            element = {}
            element['id'] = i
            if i1 < df1.shape[0] and i == df1.at[i1, 'id']:
                element['question'] = df1.at[i1, 'question']
                element['answer'] = df1.at[i1, 'answer']
                i1 += 1
            if i2 < df2.shape[0] and i == df2.at[i2, 'id']:
                element['question'] = df2.at[i2, 'question']
                element['answer'] = df2.at[i2, 'answer']
                i2 += 1
            i += 1
            data.append(element)
        except:
            break
    #print(data)
    with open(project_config.answer_path, 'w', encoding='utf-8') as file:
        for record in data:
            #print(record)
            json_record = json.dumps(record, ensure_ascii=False)
            file.write(json_record)
            file.write('\n')
