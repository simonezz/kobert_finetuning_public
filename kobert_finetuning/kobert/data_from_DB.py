import pymysql
import pandas as pd
from tqdm import tqdm
import re
import sys
import ray
import psutil


@ray.remote
def text_parsing(url):

    import sys

    sys.path.append("../../../BERT-pytorch/bert_pytorch/preprocessing")

    sys.path.append("../../../Rec_sys/utils")

    from hwpmath2latex import hwp_parser

    from preprocess import mapping_from_latex

    txt = hwp_parser(url)

    if txt == None:
        raise Exception("No text!")

    else:

        txt = txt.strip()

        txt = mapping_from_latex(txt)

        p = re.compile("\\t")

        while p.search(txt):
            txt = re.sub(p, "", txt)

        txt = " ".join(txt.split("\n"))  # one line으로 만들기

        return txt


if __name__ == "__main__":

    """
    big chapter id 205, 206, 207, ... -> 0, 1, 2...
    """
    import pickle

    with open("../../big_chapter_list.pickle", "rb") as f:
        data = pickle.load(f)

    prob_db = pymysql.connect(
        #****
    )

    curs = prob_db.cursor(pymysql.cursors.DictCursor)

    sql = """
    select p.id, p.problem_concept_id, p.url, pccc.big_chapter_id from problem p
    join problem_curriculum_concept_cache pccc
    on pccc.relation_id = p.problem_concept_id
    where pccc.revision_name="교육과정 15"
    """

    curs.execute(sql)

    df = pd.DataFrame(curs.fetchall())

    df = df.dropna(axis=0)

    num_logical_cpus = psutil.cpu_count()

    ray.init(num_cpus=num_logical_cpus)

    with open("problem_data.txt", "w") as f:

        for i in tqdm(list(df.index)):

            try:

                hwp_url = (
                    "https://***"
                    + df.loc[i, "url"]
                    + "p.hwp"
                )
                hwp_url = hwp_url.replace("math_problems/ng", "math_problems/hwp")

                f.write(
                    "\t".join(
                        [
                            str(df.loc[i, "id"]),
                            ray.get(text_parsing.remote(hwp_url)),
                            str(data.index(int(df.loc[i, "big_chapter_id"]))),
                        ]
                    )
                )
                f.write("\n")

            except:
                print(hwp_url, "이 유효하지 않지 않거나 한글이 들어있지 않음")
                pass
