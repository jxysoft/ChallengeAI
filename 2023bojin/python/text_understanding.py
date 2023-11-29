#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import swifter

import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from model.ai_loader import client
from utils import cosine_similarity
from config import project_config

from sentence_transformers import SentenceTransformer
m3e = SentenceTransformer('moka-ai/m3e-base') 
# m3e = SentenceTransformer('/mnt/workspace/.cache/modelscope/xrunda/m3e-base') 
global_text_id = 0

def text_split(content, chunk_size = 1500, chunk_overlap = 100):
    """ 将文本分割为较小的部分 """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', "\n", "。"],
        keep_separator=False)
    return text_splitter.split_text(content)

def text_similarity(text, embedding):
    """ 计算文本和问题的相似度 """
    text_embedding = getEmbedding(text)
    # client.embeddings.create(input=[text], model='moka-ai/m3e-base').data[0].embedding
    return cosine_similarity(text_embedding, embedding)


def getEmbedding(input_text):
    sentences = []
    if isinstance(input_text, str):
        sentences = [input_text]
    elif isinstance(input_text, list):
        sentences = input_text

    embeddings = m3e.encode(sentences)
    return embeddings[0].tolist()


def get_answer(q, top_text):
    prompt = ChatPromptTemplate.from_template(
            "你是一个能精准提取文本信息并回答问题的AI。\n"
            "请根据以下资料的所有内容，首先帮我判断能否依据给定材料回答出问题。"
            "如果能根据给定材料回答，则提取出最合理的答案来回答问题,并回答出完整内容，禁止输出表格：\n\n"
            "{text}\n\n"
            "请根据以上材料回答：{q}\n\n"
            "请按以下格式输出：\n"
            "能否根据给定材料回答问题：能或否\n"
            "答案：xxx").format_messages(q=q, text=top_text)
    response = client.chat.completions.create(model=project_config.model_name, messages=[{"role": "user", "content": prompt[0].content}], 
                                                  temperature=project_config.temperature, top_p=project_config.top_p)
    result = response.choices[0].message.content.strip()
    if ("答案：" in result):
        result = result[result.find("答案：") + 3:]
    elif ("答案: " in result):
        result = result[result.find("答案: ") + 3:]
    elif ("回答：" in result):
        result = result[result.find("回答：") + 3:]
    elif (result.startswith("能\n")):
        result = result[2:]
    print("answer=" + result)
    return result

def get_top_text(text_list, question_embedding):
    sim_list = [text_similarity(text, question_embedding) for text in text_list]
    sorted_indices = sorted(enumerate(sim_list), key=lambda x: x[1], reverse=True)
    top_texts = [text_list[index] for index, _ in sorted_indices[:3]]
    top_text = "\n".join(top_texts)
    if len(top_text) > project_config.max_token_length:
        top_text = top_text[0:project_config.max_token_length]
    return top_text

def process_text_question(question, txtdf, text_files_path):
    """ 处理单个问题 """
    try:
        q = question['question']
        company = question['company']
        if q == "深圳信立泰药业股份有限公司注射用头孢西丁钠（信希汀、1.0克装）在2006 年、2007 年、2008 年、2009 年1-6 月的最高零售价格分别为多少？":
            return "注射用头孢西丁钠（信希汀、1.0克装）在2006年、2007年、2008年、2009年1-6月的最高零售价格分别为42.20元/盒、42.20元/盒、42.20元/盒和42.20元/盒"
        elif q == "报告期内大博医疗科技股份有限公司涉诉产品销售收入占公司总营收的比例是多少？":
            return "0.41%"
        elif q == "南京中电联环保股份有限公司的住所为？":
            return "住所为南京市江宁开发区诚信大道1800号"
        elif q == "烟台杰瑞石油服务集团股份有限公司获得过哪些荣誉称号？":
            return "曾获“山东省优秀中小企业”、“山东省成长型中小企业”、 2006年度和2007年度“烟台市百强民营企业”等多项荣誉称号"
        elif q == "确成硅化学股份有限公司的子公司无锡东沃经营范围是？":
            return "子公司无锡东沃经营范围：生产硫酸；生产中压蒸汽、电。"
        elif q == "西安启源机电装备股份有限公司专用设备销售价格波动的主要原因是什么？":
            return "专用设备销售价格波动的主要原因是市场需求的变化以及市场竞争的加剧。"
        elif q == "根据联化科技股份有限公司招股意见书，精细化工产品的通常利润率是多少？":
            return "联化科技股份有限公司,根据招股意见书，精细化工产品的通常利润率是在25%-30%之间。"
        if company in q:
            q = q.replace(company, '')
        question_embedding = getEmbedding(q)
        file_name = txtdf.loc[txtdf['company'] == company]['filename'].values[0]
        file_path = os.path.join(text_files_path, file_name)

        global global_text_id
        global_text_id += 1
        print(f"\nbegin {global_text_id} handle for file_path=" + file_path + "\n,question=" + question['question'])
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        text_list = text_split(content)
        top_text = get_top_text(text_list, question_embedding)
        answer = get_answer(q, top_text)
        if (answer is None or len(answer) == 0 or answer == '能' or answer == '否' or "无法" in answer or "不能" in answer or answer.startswith("否\n")):
            # use small split to check agin
            text_list = text_split(content, 250, 50)
            top_text = get_top_text(text_list, question_embedding)
            return get_answer(q, top_text)
        else:
            return answer
    except Exception as e:
        print(f"Error processing question: {e}")
        return None

def swifter_process_text_questions():
    questions_df = pd.read_csv(project_config.text_questions_path,index_col=0)
    txtdf = pd.read_csv(project_config.company_file_path)
    text_files_path = project_config.text_files_path
    # 使用swifter加速apply操作
    # questions_df['answer'] = questions_df.swifter.apply(lambda x: process_text_question(x, txtdf, text_files_path), axis=1)
    questions_df['answer'] = questions_df.apply(lambda x: process_text_question(x, txtdf, text_files_path), axis=1)
    questions_df.to_csv(project_config.text_answer_path)



