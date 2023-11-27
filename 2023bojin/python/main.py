#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('/your/file/path')

from processing import extract_company_names, split_questions_by_type,integrate_outputs
from text_understanding import swifter_process_text_questions
from data_extraction import swifter_process_data_questions
import pandas as pd
from config import project_config


def main():
    # 公司名提取
    extract_company_names()
    # 问题类型分类
    split_questions_by_type()
    # 文本理解
    swifter_process_text_questions()
    # 数据查询
    swifter_process_data_questions()
    # 答案合并
    integrate_outputs()
if __name__ == "__main__":
    main()
