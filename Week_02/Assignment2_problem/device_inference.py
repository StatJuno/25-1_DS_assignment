import argparse
import json
import torch
import pprint
from sympy import false
from tqdm import tqdm
import os
import torch.distributed as dist
import re
from itertools import chain
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.multiprocessing as mp
from utils import (
    load_model_and_tokenizer,
    create_incorrect_loader,
    create_data_loader,
    init_distributed,
    cleanup_distributed,
    save_log_arguments,
    warn_truncation,
)

pp = pprint.PrettyPrinter(indent=2).pprint

# 먼저 인자 파싱 함수 정의
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument("--task", type=str, default="cqa", help="Which dataset to run on")
    parser.add_argument("--exp_iter", type=int, default=-1, help="exp iteration for logging")
    parser.add_argument("--seed", type=int, default=10, help="Random seed for shuffling data (default: 10)")
    parser.add_argument("--log_dir", type=str, default="", help="logging dir")
    return parser.parse_args()


# 데이터 예시를 파일에 기록하는 함수
def write_output_example(config, out_path, prediction, record, eos_token):
    question_data = record["question"]
    question_text = question_data["stem"]
    options = question_data["choices"]
    options_str = "\n".join([f'({option["label"]}) {option["text"]}' for option in options])
    formatted_example = f"Q: {question_text}\nOptions:\n{options_str}\nA: {prediction}" + eos_token

    with open(out_path, "a+") as out_file:
        print(formatted_example, file=out_file, end="\n\n")

    return formatted_example


# 예측 결과를 평가하는 함수 (정답, 오답 처리)
def compute_metric(config, preds, dataset, out_path, tok, show_hint):
    incorrect_items = []
    correct_count, total_count = 0, 0
    try:
        for idx, (pred_item, record) in enumerate(zip(preds, dataset), 1):
            try:
                is_correct = False
                correct_answer = record.get("answer")
                if correct_answer is None:
                    print(f"Warning: Missing answer for index {idx}")
                    continue

                marker_index = pred_item.find("Q: ")
                if marker_index != -1:
                    pred_item = pred_item[:marker_index]

                if "####" in pred_item:
                    parts = pred_item.split("####")
                    if len(parts) > 1 and len(parts[1].split()) > 0:
                        pred_item = parts[0] + "#### " + parts[1].split()[0]
                    else:
                        pred_item = parts[0] + "#### "

                extracted_ans = None
                try:
                    matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred_item))
                    extracted_ans = matches[-1].group(1) if matches else None
                except IndexError as exc:
                    print(f"Warning: Failed to extract answer from prediction at index {idx}: {exc}")
                    continue

                if extracted_ans and extracted_ans == correct_answer:
                    is_correct = True

                if is_correct:
                    correct_count += 1
                    try:
                        write_output_example(config, out_path + "/correct_data.txt", pred_item, record, tok.eos_token)
                    except Exception as exc:
                        print(f"Warning: Failed to write output example at index {idx}: {exc}")
                else:
                    if not show_hint:
                        incorrect_items.append(record)
                total_count += 1

            except Exception as inner_exc:
                print(f"Warning: Error processing prediction at index {idx}: {inner_exc}")
                continue

    except Exception as exc:
        print(f"Critical error in compute_metric: {exc}")

    return incorrect_items, correct_count, total_count


# 입력 예시를 토크나이징하는 함수 (프롬프트 결합)
def prepare_prompts(config, examples, tok, prompt_text, show_hint):
    combined_queries = []
    for question_item, answer_key in zip(examples["question"], examples["answerKey"]):
        q_text = question_item["stem"]
        choices_list = question_item["choices"]
        options_formatted = "\n".join([f'({choice["label"]}) {choice["text"]}' for choice in choices_list])
        if show_hint:
            combined_queries.append(f"{prompt_text}\nQ: {q_text} ({answer_key})\nOptions:\n{options_formatted}\nA: ")
        else:
            combined_queries.append(f"{prompt_text}\nQ: {q_text}\nOptions:\n{options_formatted}\nA: ")

    tokenized_queries = tok(
        combined_queries,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
    )

    warn_truncation(config, combined_queries, tok, log_point="Data generation")

    return tokenized_queries


# 배치 단위로 평가를 수행하는 함수
def evaluate_batches(config, mdl, device_rank, loader, tok, generation_length, prompt_str, show_hint=False):
    generate_func = mdl.module.generate if hasattr(mdl, "module") else mdl.generate

    progress_bar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"{'Hint' if show_hint else 'No Hint'} Eval [Rank {device_rank}]",
        position=device_rank + 1,
        leave=False,
        disable=(device_rank != 0),
    )

    total_correct = 0
    overall_total = 0
    incorrect_records = []

    with torch.no_grad():
        for batch_idx, batch_data in progress_bar:
            try:
                tokenized_batch = prepare_prompts(config, batch_data, tok, prompt_str, show_hint=show_hint)
                input_ids = tokenized_batch["input_ids"].to(device_rank)
                attention_mask = tokenized_batch["attention_mask"].to(device_rank)

                try:
                    outputs = generate_func(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.size(1) + generation_length,
                        pad_token_id=tok.eos_token_id,
                        do_sample=True,
                        top_p=0.9,
                        temperature=1.0,
                    )
                except Exception as gen_exc:
                    print(f"Warning: Generation failed for batch {batch_idx}: {gen_exc}")
                    continue

                try:
                    generated_tokens = outputs[:, input_ids.shape[-1] :]
                    decoded_preds = tok.batch_decode(generated_tokens, skip_special_tokens=True)
                except Exception as dec_exc:
                    print(f"Warning: Decoding failed for batch {batch_idx}: {dec_exc}")
                    continue

                gathered_preds = [None for _ in range(dist.get_world_size())]
                gathered_data = [None for _ in range(dist.get_world_size())]

                try:
                    dist.all_gather_object(gathered_preds, decoded_preds)
                    dist.all_gather_object(gathered_data, batch_data)
                except Exception as ag_exc:
                    print(f"Warning: All-gather failed for batch {batch_idx}: {ag_exc}")
                    continue

                if device_rank == 0:
                    try:
                        merged_preds = list(chain.from_iterable(gathered_preds))
                        merged_dataset = []
                        for proc_data in gathered_data:
                            for i in range(len(proc_data["question"])):
                                single_record = {}
                                for key in proc_data.keys():
                                    single_record[key] = proc_data[key][i]
                                merged_dataset.append(single_record)

                        incorrect_batch, batch_correct, batch_total = compute_metric(
                            config, merged_preds, merged_dataset, config.target_save, tok, show_hint=show_hint
                        )
                        total_correct += batch_correct
                        overall_total += batch_total
                        if not show_hint:
                            incorrect_records.extend(incorrect_batch)
                    except Exception as merge_exc:
                        print(f"Warning: Processing results failed for batch {batch_idx}: {merge_exc}")
                        continue

                dist.barrier()

            except Exception as batch_exc:
                print(f"Warning: Failed to process batch {batch_idx}: {batch_exc}")
                continue

    if device_rank == 0:
        if overall_total > 0:
            if show_hint:
                print(f"Hint Correct: {total_correct}, Accuracy: {total_correct / overall_total:.4f}")
            else:
                print(f"No hint Correct: {total_correct}, Accuracy: {total_correct / overall_total:.4f}")
        else:
            print("Warning: No valid examples were processed")

    return incorrect_records, total_correct, overall_total


# 리스트를 분산 환경에서 broadcast하는 함수
def distribute_list(data, src_rank):
    obj_list = [data if dist.get_rank() == src_rank else None]
    dist.broadcast_object_list(obj_list, src=src_rank)
    return obj_list[0]


# 전체 평가를 수행하는 함수 (두 번 평가 진행)
def run_evaluation(config, mdl, device_rank, total_devices, loader, tok, generation_length, out_path, prompt_str, prompt_hint_str):
    mdl.eval()
    incorrect_records, correct_count, total_count = evaluate_batches(config, mdl, device_rank, loader, tok, generation_length, prompt_str, show_hint=False)
    incorrect_records = distribute_list(incorrect_records, src_rank=0)
    wrong_loader, sampler_wrong = create_incorrect_loader(config, incorrect_records, device_rank, total_devices)
    wrong_records, additional_correct, additional_total = evaluate_batches(config, mdl, device_rank, wrong_loader, tok, generation_length, prompt_str, show_hint=False)
    correct_count += additional_correct
    hint_correct, hint_total = "_", "_"
    dist.barrier()

    return correct_count, total_count, hint_correct, hint_total


# 분산 학습 환경을 위한 메인 함수
def distributed_main(rank, world_size, config):
    init_distributed(rank, world_size)
    torch.cuda.set_device(rank)

    # 프롬프트 설정
    base_prompt = ""
    hint_prompt = ""
    base_prompt_path = f"./n_shot_prompts/{config.task}.json"
    hint_prompt_path = f"./n_shot_prompts/{config.task}_hint.json"
    with open(base_prompt_path, "r") as fp:
        base_data = json.load(fp)
    with open(hint_prompt_path, "r") as fp:
        hint_data = json.load(fp)
    base_prompt_list = [item["prompt"] for item in base_data["n_shot_prompts"]]
    hint_prompt_list = [item["prompt"] for item in hint_data["n_shot_prompts"]]
    base_prompt = "\n".join(base_prompt_list)
    hint_prompt = "\n".join(hint_prompt_list)

    model, tok = load_model_and_tokenizer(config, config.model_name, rank, eval_mode=True)

    tokenized_base = tok(base_prompt, return_tensors="pt")
    base_len = tokenized_base["input_ids"].shape[1]
    tokenized_hint = tok(hint_prompt, return_tensors="pt")
    hint_len = tokenized_hint["input_ids"].shape[1]
    config.max_length += max(base_len, hint_len)

    config.batch_size = config.test_batch_size  # inference 시 배치 사이즈

    train_loader, sampler_train = create_data_loader(config, tok, rank, world_size)

    corr, tot, corr_hint, tot_hint = run_evaluation(config, model, rank, world_size, train_loader, tok, config.gen_length, config.target_save, base_prompt, hint_prompt)
    if rank == 0:
        accuracy = corr / tot
        hint_accuracy = "_"
        print(f" {config.task}, accuracy: {accuracy}, hint_accuracy: {hint_accuracy}")
        save_log_arguments(f"{config.log_dir}/eval_log.json", iter=config.exp_iter, accuracy=accuracy, hint_accuracy=hint_accuracy)

    cleanup_distributed()


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    params = json.load(open(args.config))

    args.batch_size = 2
    args.test_batch_size = 32
    args.model_name = params["model_name"]
    args.precision = "bf16"
    args.max_length = params["max_length"]
    args.gen_length = params["gen_length"]
    args.n_shot = 7

    # STaR specific
    args.name = params["name"]
    args.target_save = params["target_save"]
    args.model_dir = params["model_dir"]
    try:  # load from trained model
        args.total_steps = params["total_steps"]
    except:
        args.total_steps = 0

    args.method = params["method"]
    torch.manual_seed(args.seed)

    num_devices = torch.cuda.device_count()
    mp.spawn(distributed_main, args=(num_devices, args), nprocs=num_devices, join=True)
