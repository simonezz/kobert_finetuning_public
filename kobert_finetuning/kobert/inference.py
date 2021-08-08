import torch
from fine_tuning import BERTClassifier
from hwpmath2latex import hwp_parser
from preprocess import mapping_from_latex


model = torch.load("freewheelin_model.pt")

model.eval()

import pymysql
import pandas as pd

prob_db = pymysql.connect(user="real", passwd="", host="", db="iclass", charset="utf8")

curs = prob_db.cursor(pymysql.cursors.DictCursor)

sql = """
select tmp.ID, tmp.problemURL, Tmc.chapter_big from Table_middle_problems as tmp
join Table_middle_contents Tmc on tmp.unitCode = Tmc.unitCode
where tmp.ID > 603927 and tmp.unitCode != -1
"""

curs.execute(sql)
df = pd.DataFrame(curs.fetchall())


d = dict()

for i, c in enumerate(classes):
    d[c] = i

import pymysql
import pandas as pd

prob_db = pymysql.connect(user="admin", passwd="", host="", db="", charset="utf8")

curs = prob_db.cursor(pymysql.cursors.DictCursor)

sql = """
select big_chapter_name, big_chapter_id from problem_curriculum_concept_cache
"""

curs.execute(sql)
new_DB_df = pd.DataFrame(curs.fetchall())

new_DB_df = new_DB_df.drop_duplicates(["big_chapter_id"])  # 대단원 이름기준으로 중복 제거
new_DB_df = new_DB_df.dropna()

for i in list(new_DB_df.index):
    try:
        new_DB_df.loc[i, "class"] = d[str(int(new_DB_df.loc[i, "big_chapter_id"]))]
    except:
        pass

new_DB_df = new_DB_df.dropna()

new_DB_df.set_index("big_chapter_name", inplace=True)

import sys
from tqdm import notebook

sys.path.append("../../../Rec_sys/utils")

sys.path.append("../../../BERT-pytorch/bert_pytorch/preprocessing")


unseen_values = []

for i in notebook.tqdm(list(df.index)):
    hwp_url = ""

    #     print(hwp_url)

    txt = mapping_from_latex(hwp_parser(hwp_url))

    unseen_values.append(
        [txt, str(int(new_DB_df.loc[df.loc[i, "chapter_big"], "class"]))]
    )


from fine_tuning import BERTDataset
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

_, vocab = get_pytorch_kobert_model()

max_len = 64

device = torch.device("cuda")

tokenizer = get_tokenizer()

tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

test_set = BERTDataset(unseen_values, 0, 1, tok, max_len, True, False)

test_input = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=5)

count = 0
true = 0

for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(
    notebook.tqdm(test_input)
):

    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length = valid_length
    out = model(token_ids, valid_length, segment_ids)

    if int(out.argmax().cpu().detach().numpy()) == int(
        unseen_values[batch_id][1]
    ):  # true
        true += 1

    else:  # false
        print(
            f"{unseen_values[batch_id][0]}, 예측 :  {out.argmax().cpu().detach().numpy()} / {list(new_DB_df[new_DB_df['class'] == int(out.argmax().cpu().detach().numpy())].index)[0]}, 실제 : {unseen_values[batch_id][1]}/{list(new_DB_df[new_DB_df['class'] == int(unseen_values[batch_id][1])].index)[0]}"
        )
        print("\n")
    count += 1

print(f"accuracy : {true / count}")
#     if out.argmax().cpu().detach().numpy() == 0:
#         print(unseen_values[batch_id][0], ": 부정")
#     else:
#         print(unseen_values[batch_id][0], ": 긍정")
