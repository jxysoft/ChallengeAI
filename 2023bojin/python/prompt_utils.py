import langchain
import time
import pandas as pd
from config import project_config
from langchain import ChatOpenAI

chat = ChatOpenAI(openai_api_base="http://127.0.0.1:8000/v1/",  openai_api_key='None')
llm = chat

from langchain.utilities import SQLDatabase
db = SQLDatabase.from_uri(project_config.db_sqlite_url, sample_rows_in_table_info=2)
# dbInfo = db.get_table_info()
# print(len(dbInfo))
# print(db.get_table_info())

from langchain.prompts.prompt import PromptTemplate

examples = [
        {
            "input": "请查询在2021年度，688338股票涨停天数？   解释：（收盘价/昨日收盘价-1）》=9.8% 视作涨停",
            "sql_cmd": """SELECT COUNT(*) AS "涨停天数"
FROM "A股票日行情表"
WHERE "股票代码" = '688338'
AND "交易日" LIKE '2021%'
AND ("收盘价(元)" / "昨收盘(元)") >= 1.098;""",
            "result": "[(1,)]",
            "answer": "涨停天数为1天",
        },
        {
            "input": "请帮我计算，在20210105，中信行业分类划分的一级行业为综合金融行业中，涨跌幅最大股票的股票代码是？涨跌幅是多少？百分数保留两位小数。股票涨跌幅定义为：（收盘价 - 前一日收盘价 / 前一日收盘价）* 100%。",
            "sql_cmd": """SELECT B."股票代码", (("收盘价(元)" - "昨收盘(元)") / "昨收盘(元)") * 100 AS "涨跌幅"
FROM "A股公司行业划分表" AS A
JOIN "A股票日行情表" AS B ON A."股票代码" = B."股票代码" AND A."交易日期" = B."交易日"
WHERE A."行业划分标准" = "中信行业分类" 
AND A."一级行业名称" = '综合金融' 
AND B."交易日" = '20210105'
ORDER BY "涨跌幅" DESC
LIMIT 1;""",
            "result": "[('600120', 0.0,)]",
            "answer": "根据查询结果显示，20210105日，中信行业分类划分的一级行业为综合金融行业中，涨跌幅最大股票的股票代码是600120,涨跌幅是0.0",
        },
        {
            "input": "帮我查询出20210415日，建筑材料一级行业涨幅超过5%（不包含）的股票数量。",
            "sql_cmd": """SELECT count(1)
FROM "A股公司行业划分表" AS A
JOIN "A股票日行情表" AS B ON A."股票代码" = B."股票代码" AND A."交易日期" = B."交易日"
WHERE A."一级行业名称" = '建筑材料' 
AND B."交易日" = '20210415'
AND (B."收盘价(元)" - B."昨收盘(元)") > 0.05;""",
            "result": "[(19,)]",
            "answer": "根据查询结果显示，20210415日，建筑材料一级行业涨幅超过5%（不包含）的股票数量为19只。",
        },
]

example_prompt = PromptTemplate(
    input_variables=["input", "sql_cmd", "result", "answer",],
    template="\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult: {result}\nAnswer: {answer}",
)

# print(example_prompt.format(**examples[0]))

from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


# embeddings = HuggingFaceEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name='/mnt/workspace/.cache/modelscope/xrunda/m3e-base')
to_vectorize = [" ".join(example.values()) for example in examples]

vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=1,
)

from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt

#print(PROMPT_SUFFIX)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=_mysql_prompt,
    suffix=PROMPT_SUFFIX, 
    input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
)

from langchain_experimental.sql import SQLDatabaseChain
local_chain = SQLDatabaseChain.from_llm(llm, db, prompt=few_shot_prompt, use_query_checker=True, 
                                        verbose=True, return_sql=False,)

def prompt_process_data_questions():
    start = 0
    questions_df = pd.read_csv(project_config.data_questions_path,index_col=0)[start:]
    output = pd.DataFrame(columns=['id', 'question', 'formatted_answer'])

    for i, row in questions_df.iterrows():
        start = time.time()
        answer = None
        print("begin handle i=" + str(i) + ", time=" + str(start) + ", question=" + row['question'])
        try:
            answer =  local_chain.run(row['question'])
        except:
            answer = "查找不到或者异常"
        output.at[i, 'id'] = row['id']
        output.at[i, 'question'] = row['question']
        output.at[i, 'formatted_answer'] = answer
        end = time.time()
        print("end handle i=" + str(i) + ",end=" + str(end) + ", time=" + str(end - start) + ", answer=" + answer)
        if (end - start) < 55:
            time.sleep(61 - (end - start))
    output.to_csv(project_config.data_answer_path)