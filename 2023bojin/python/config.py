#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
def get_base_file_path():
    return os.path.dirname(os.path.dirname(__file__))

class Config:
    base_path = get_base_file_path()
    data_path = os.path.join(base_path, "data")
    text_files_path = os.path.join(data_path, "txt").replace("\\", "/")
    db_sqlite_url = os.path.join(data_path, "db/bojin.db").replace("\\", "/")

    save_path = os.path.join(data_path, "stock_descriptions/txtfile2company.csv").replace("\\", "/")

    company_file_path = os.path.join(data_path, "stock_descriptions/txtfile2company.csv").replace("\\", "/")
    text_questions_path = os.path.join(data_path, "sample_data/文本理解题.csv").replace("\\", "/")
    data_questions_path = os.path.join(data_path, "sample_data/数据查询题.csv").replace("\\", "/")

    text_answer_path = os.path.join(data_path, "sample_data/文本理解题答案.csv").replace("\\", "/")
    data_answer_path = os.path.join(data_path, "sample_data/数据查询题答案.csv").replace("\\", "/")

    questions_path = os.path.join(base_path, "input/question.json").replace("\\", "/")
    answer_path = os.path.join(base_path, "output/answer.jsonl")
    max_token_length = 3000
    model_name = "Tongyi-Finance-14B-Chat"
    temperature = 0.6
    top_p = 0.5

project_config = Config()
