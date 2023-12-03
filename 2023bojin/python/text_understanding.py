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
m3e = None
global_text_id = 0


def get_text_splitter(chunk_size = 1500, chunk_overlap = 100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', "\n", "。"],
        keep_separator=False)
    return text_splitter

def text_split(content, chunk_size = 1500, chunk_overlap = 100):
    """ 将文本分割为较小的部分 """
    text_splitter = get_text_splitter(chunk_size, chunk_overlap)
    return text_splitter.split_text(content)

def text_similarity(text, embedding):
    """ 计算文本和问题的相似度 """
    text_embedding = getEmbedding(text)
    # client.embeddings.create(input=[text], model='moka-ai/m3e-base').data[0].embedding
    return cosine_similarity(text_embedding, embedding)

def getEmbeddingModel():
    global m3e
    if m3e is None:
        m3e = SentenceTransformer(project_config.embedding_model_name) 
    return m3e

def getEmbedding(input_text):
    sentences = []
    if isinstance(input_text, str):
        sentences = [input_text]
    elif isinstance(input_text, list):
        sentences = input_text

    embeddings = getEmbeddingModel().encode(sentences)
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


###### test vector store
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

vector_dbs = {}
small_vector_dbs = {}
def init_vectorstore():
    embedding_function = SentenceTransformerEmbeddings(model_name=project_config.embedding_model_name) 
    text_files_path = project_config.text_files_path
    txtdf = pd.read_csv(project_config.company_file_path)
    text_splitter = get_text_splitter(1500, 100)
    small_text_splitter = get_text_splitter(250, 50)
    for i, row in txtdf.iterrows():
        company = row['company']
        file_name = row['filename']
        file_path = os.path.join(text_files_path, file_name)
        loader = TextLoader(file_path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        small_docs = small_text_splitter.split_documents(documents)
        print("init vectorstore for company=" + company + ", file_path=" + file_path)
        # load it into Chroma
        db = Chroma.from_documents(docs, embedding_function)
        vector_dbs[company] = db
        small_db = Chroma.from_documents(small_docs, embedding_function)
        vector_dbs[company] = small_db
    print("init vectorstore done len=" + str(len(vector_dbs)))
  
def vector_get_top_text(db, question, top_k = 3):
    print("begin vector_get_top_text for question=" + question)
    docs = db.similarity_search(question,k = top_k)
    top_texts = [doc.page_content for doc in docs]
    top_text = "\n".join(top_texts)
    if len(top_text) > project_config.max_token_length:
        top_text = top_text[0:project_config.max_token_length]
    print("top_text=" + top_text)
    return top_text
  
def vector_process_text_question(question):
    """ 处理单个问题 """
    try:
        q = question['question']
        company = question['company']
        if company in q:
            q = q.replace(company, '')
        vector_db = vector_dbs[company]
        small_db = small_vector_dbs[company]
        global global_text_id
        global_text_id += 1
        print(f"\nbegin {global_text_id} handle for company=" + company + "\n,question=" + question['question'])
        
        top_text = vector_get_top_text(vector_db, q, 3)
        answer = get_answer(q, top_text)
        if (answer is None or len(answer) == 0 or answer == '能' or answer == '否' or "无法" in answer or "不能" in answer or answer.startswith("否\n")):
            # use small split to check agin
            top_text = vector_get_top_text(small_db, q, 10)
            return get_answer(q, top_text)
        else:
            return answer
    except Exception as e:
        print(f"Error processing question: {e}")
        return None

