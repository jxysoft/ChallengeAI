# 需要能处理表格和异常的pdf转换，image没关注
# first PyPDFLoader ， then PyMuPDFLoader
# ! pip install pypdf PyMuPDF

import re
pattern = r'(.*招股意向书)?[\s\n]+.*[\d-]+[\s\n]*'

from langchain.document_loaders import *

import os
pdfpath = "/Users/jxy/Projects/person/ChallengeAI/2023bojin/data/pdf/"
txtpath = "/Users/jxy/Projects/person/ChallengeAI/2023bojin/data/newtxt/"
filepaths = os.listdir(pdfpath)
filepaths.sort()
print("total file="  + str(len(filepaths)))
for idx, filename in enumerate(filepaths):
    print("begin i=" + str(idx) + " filename= " + filename)
    flagSub = False
    with open(txtpath + filename.replace("PDF", "txt"), "w") as out:
        try:
            docs = PyPDFLoader("/Users/jxy/Projects/person/ChallengeAI/2023bojin/data/pdf/" + filename).load()
        except Exception as e:
            print("error i=" + str(idx) + " filename= " + filename)
            docs = PyMuPDFLoader("/Users/jxy/Projects/person/ChallengeAI/2023bojin/data/pdf/" + filename).load()
        for doc in docs:
            # only for head
            max_len = len(doc.page_content)
            head = ""
            rest = ""
            if doc.metadata['page'] != 0: # 忽略第一页
                if max_len > 150:
                    head = doc.page_content[0:150]
                    rest = doc.page_content[150:]
                else:
                    head = doc.page_content
                res_head = re.sub(pattern, "", head, flags=re.DOTALL)
                if len(res_head) != len(head):
                    flagSub = True
                out.write(res_head)
                out.write(rest)
            else:
                out.write(doc.page_content)
        print("end i =" + str(idx) + ", filename= " +filename + " flagSub=" + str(flagSub))
print("end")