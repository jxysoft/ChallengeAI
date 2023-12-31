#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
def get_base_file_path():
    return os.path.dirname(os.path.dirname(__file__))

import sys
sys.path.append(get_base_file_path())

from processing import extract_company_names, split_questions_by_type,integrate_outputs
from text_understanding import swifter_process_text_questions, vector_process_text_question
from data_extraction import swifter_process_data_questions
from prompt_utils import prompt_process_data_questions
import pandas as pd
from config import project_config
from pdf_convert import convert_pdf

def main():
    # pdf handle
    convert_pdf()
    # 公司名提取
    extract_company_names()
    # 问题类型分类
    split_questions_by_type()
    # 文本理解
    swifter_process_text_questions()
    #vector_process_text_question
    # 数据查询
    # swifter_process_data_questions()
    # langchain + few shot
    prompt_process_data_questions()
    # 答案合并
    integrate_outputs()
if __name__ == "__main__":
    main()
