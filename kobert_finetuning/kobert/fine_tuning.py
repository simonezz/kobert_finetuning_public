# Fine-Tuning with custom data
import argparse
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import matplotlib.pyplot as plt

import os, re

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import wandb

import glob

from utils.torch_utils import ModelEMA

# 1번 GPU 사용 시
device = torch.device("cuda:1")


def increment_path(path, exist_ok=True, sep=""):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


# accuracy 계산
def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


# BERT Dataset정의(변환과정 포함)


class BERTDataset(Dataset):
    def __init__(
        self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair
    ):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair
        )

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return self.sentences[i] + (self.labels[i],)

    def __len__(self):
        return len(self.labels)


class BERTClassifier(nn.Module):
    def __init__(
        self, bert, hidden_size=768, num_classes=136, dropout=None, params=None
    ):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dropout = dropout

        self.classifier = nn.Linear(hidden_size, num_classes)
        # self.classifier = nn.Sequential(nn.Linear(hidden_size, 400), nn.ReLU(), nn.Linear(400, num_classes)) # hidden layer 추가

        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):

        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        # return_dict = False 설정해주어야 오류가 안남.
        _, pooler = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.float().to(token_ids.device),
        )
        #         print(pooler)
        if self.dropout:
            out = self.dropout(pooler)

        return self.classifier(out)


if __name__ == "__main__":

    # Pre-trained KoBERT 불러오기
    bertmodel, vocab = get_pytorch_kobert_model()

    # KoBERT 형태소 분석기
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    ## 파라미터 설정

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=64, help="maximum length")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.5, help="drop_rate")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--weight", default="kobert_from_pretrained/pytorch_model.bin")
    parser.add_argument("--name", default="kobert_finetuning")
    parser.add_argument("--evolve", action="store_true", help="evolve hyperparameters")
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )

    opt = parser.parse_args()

    opt.save_dir = increment_path(
        Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve
    )

    opt.global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1

    rank = opt.global_rank
    # max_len = 64
    # batch_size = 64
    # warmup_ratio = 0.2
    # epochs = 20
    # max_grad_norm = 1
    # log_interval = 500
    # learning_rate =  5e-5
    # dropout = 0.4
    ckpt = torch.load(opt.weight, map_location=device)
    if wandb:

        wandb_run = wandb.init(
            config=opt,
            resume="allow",
            project="KoBERT-fineTuning"
            if opt.project == "runs/train"
            else Path(opt.project).stem,
            id=ckpt.get("wandb_id") if "ckpt" in locals() else None,
        )

    # thres = [20, 22, 24, 26, 28, 30]
    thres = [30]
    for thr in thres:
        """
        custom data 사용

        """
        dataset_train = nlp.data.TSVDataset(
            f"tuning_1/prob_sol_{thr}_train.txt", field_indices=[1, 2]
        )
        dataset_test = nlp.data.TSVDataset(
            f"tuning_1/prob_sol_{thr}_test.txt", field_indices=[1, 2]
        )

        # train, test set 생성

        data_train = BERTDataset(dataset_train, 0, 1, tok, opt.max_len, True, False)
        data_test = BERTDataset(dataset_test, 0, 1, tok, opt.max_len, True, False)

        train_dataloader = torch.utils.data.DataLoader(
            data_train, batch_size=opt.batch_size, num_workers=5
        )
        test_dataloader = torch.utils.data.DataLoader(
            data_test, batch_size=opt.batch_size, num_workers=5
        )

        print(f"data 로딩 완료. soluion data thres :{thr}")

        # dropout=0.5로 BERT classifier 정의

        model = BERTClassifier(bertmodel, dropout=opt.dropout).to(device)

        # EMA
        ema = ModelEMA(model) if rank in [-1, 0] else None

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=opt.learning_rate
        )  # Adam optimizer with weight decay
        loss_fn = nn.CrossEntropyLoss()

        t_total = len(train_dataloader) * opt.epochs
        warmup_step = int(t_total * opt.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total
        )

        print("학습 시작")
        # Training

        from tqdm import notebook

        train_accuracies = []
        test_accuracies = []

        start_epoch, best_fitness = 0, 0.0

        for e in range(opt.epochs):
            train_acc = 0.0
            test_acc = 0.0
            model.train()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(
                notebook.tqdm(train_dataloader)
            ):
                optimizer.zero_grad()
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length = valid_length
                label = label.long().to(device)

                out = model(token_ids, valid_length, segment_ids)
                loss = loss_fn(out, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                train_acc += calc_accuracy(out, label)
                if batch_id % opt.log_interval == 0:
                    print(
                        "epoch {} batch id {} loss {} train acc {}".format(
                            e + 1,
                            batch_id + 1,
                            loss.data.cpu().numpy(),
                            train_acc / (batch_id + 1),
                        )
                    )
            print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
            train_accuracies.append(train_acc / (batch_id + 1))
            model.eval()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(
                notebook.tqdm(test_dataloader)
            ):
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length = valid_length
                label = label.long().to(device)
                out = model(token_ids, valid_length, segment_ids)
                test_acc += calc_accuracy(out, label)
            print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))

            test_accuracies.append(test_acc / (batch_id + 1))

            # Log
            tags = ["train/accuracy", "test/accuracy"]  # params

            for x, tag in zip([train_accuracies[-1]] + [test_accuracies[-1]], tags):

                if wandb:
                    wandb.log({tag: x})  # W&B

        results_file = opt.save_dir + "/results.txt"

        try:
            os.makedirs(opt.save_dir)
        except:
            pass

        # with open(results_file, 'r') as f:  # create checkpoint
        #     ckpt = {'epoch': opt.num_epoch,
        #             'training_results': f.read(),
        #             'model': ema.ema,
        #             'wandb_id': wandb_run.id if wandb else None}
        # 모델 저장
        model_name = f"freewheelin_{opt.epochs}_warmup_{opt.warmup_ratio}_dropout_{opt.dropout}_{thr}"

        model_save_path = f"./weights/{model_name}.pt"

        torch.save(model, model_save_path)

        print("model saved at ", model_save_path)

        plt.plot(list(range(1, opt.epochs + 1)), train_accuracies)
        plt.plot(list(range(1, opt.epochs + 1)), test_accuracies)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.legend(["train", "test"])

        plt.savefig(f"{model_name}.png")
