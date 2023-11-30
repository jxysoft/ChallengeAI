
from langchain.prompts import ChatPromptTemplate

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

db = SQLDatabase.from_uri("sqlite:///"+ project_config.db_sqlite_url, sample_rows_in_table_info=2)
def get_schema(_):
    return db.get_table_info()
def run_query(query):
    return db.run(query)

from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

model = ChatOpenAI()

sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

sql_response.invoke({"question": "请查询在2021年度，688338股票涨停天数？   解释：（收盘价/昨日收盘价-1）》=9.8% 视作涨停"})