import re
# 页眉页脚
pattern = r'[\s\n]*(.*招\s?股\s?意向书.*)?[\s\n]+[\d-]*[\s\n]*'

from langchain.document_loaders import *

import os


def convert_pdf():
    pdfpath = "/Users/jxy/Projects/person/ChallengeAI/2023bojin/data/pdf/"
    txtpath = "/Users/jxy/Projects/person/ChallengeAI/2023bojin/data/newtxt/"
    filepaths = os.listdir(pdfpath)
    filepaths.sort()
    print("total file="  + str(len(filepaths)))
    for idx, filename in enumerate(filepaths):
        print("begin i=" + str(idx) + " filename= " + filename)
        flagSub = False
        # first PyPDFLoader ， then PyMuPDFLoader
        with open(txtpath + filename.replace("PDF", "txt"), "w") as out:
            try:
                if filename == "07d29cd67ca8e0fc932e05178db1fcdca1cee937.PDF": # 繁体字
                    docs = PyMuPDFLoader("/Users/jxy/Projects/person/ChallengeAI/2023bojin/data/pdf/" + filename).load()
                else:
                    docs = PyPDFLoader("/Users/jxy/Projects/person/ChallengeAI/2023bojin/data/pdf/" + filename).load()
            except Exception as e:
                print("error i=" + str(idx) + " filename= " + filename)
                docs = PyMuPDFLoader("/Users/jxy/Projects/person/ChallengeAI/2023bojin/data/pdf/" + filename).load()
            for doc in docs:
                match = re.match(pattern, doc.page_content)
                if match and match.end() < 100:
                    out.write(doc.page_content[match.end():]) # ignore page header
                else:
                    out.write(doc.page_content)
            print("end i =" + str(idx) + ", filename= " +filename + " flagSub=" + str(flagSub))
    print("end")

# 有些文件第一页读不出来