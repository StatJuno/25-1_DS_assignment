import os
import json
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import MixedPrecision, FullStateDictConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# ------------------------- Logging 관련 함수 -------------------------

def save_log_arguments(log_path, **kwargs):
    """
    기존 로그 파일을 불러오거나 새 로그 목록을 생성하여
    새로운 인자를 추가 후 파일에 저장합니다.
    """
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as jf:
                logs = json.load(jf)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
    else:
        logs = []
    logs.append(kwargs)
    with open(log_path, "w") as jf:
        json.dump(logs, jf, indent=4)

def warn_truncation(cfg, prompt_list, tokenizer, log_point):
    """
    각 프롬프트의 원본 토큰 길이가 최대 길이를 초과하는지 검사하고,
    초과한 경우 경고 로그를 남깁니다.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s - %(message)s",
        handlers=[logging.FileHandler(f"{cfg.log_dir}/truncation_warnings.log")]
    )
    orig_lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompt_list]
    is_truncated = [orig > cfg.max_length for orig in orig_lengths]

    if any(is_truncated):
        num_truncated = sum(is_truncated)
        logging.warning(f"{num_truncated} input(s) exceeded max_length and were truncated.")
        for idx, (prompt, length, truncated_flag) in enumerate(zip(prompt_list, orig_lengths, is_truncated)):
            if truncated_flag:
                logging.warning(
                    f"Iteration: {cfg.exp_iter}, Log point: {log_point}\n"
                    f"--Truncated Prompt {idx+1} (Token Length {length} > Max Length {cfg.max_length})--\n{prompt}"
                )

# ------------------------- 데이터 전처리 및 Collation -------------------------

def preprocess_data(cfg, examples, tokenizer, split_mode):
    """
    질문과 선택지를 합쳐 텍스트로 구성한 후 토크나이징합니다.
    train 모드인 경우 정답도 함께 포함합니다.
    """
    merged_texts = []
    for question_item, answer_item in zip(examples["question"], examples["answerKey"]):
        q_text = question_item['stem']
        choices = question_item['choices']
        opts = "\n".join([f'({choice["label"]}) {choice["text"]}' for choice in choices])
        if split_mode == "train":
            merged_texts.append(f"Q: {q_text}\nOptions:\n{opts}\nA: {answer_item}")
        else:
            merged_texts.append(f"Q: {q_text}\nOptions:\n{opts}\nA: ")
    
    tokenized_data = tokenizer(
        merged_texts,
        padding="max_length",
        truncation=True,
        max_length=cfg.max_length
    )
    tokenized_data["question"] = examples["question"]
    tokenized_data["answer"] = examples["answerKey"]

    warn_truncation(cfg, merged_texts, tokenizer, log_point="Simple data load")
    return tokenized_data

def custom_collate(batch):
    """
    배치 내 각 항목의 텐서 데이터는 torch.tensor로 변환하고,
    문자열 등 텐서가 아닌 데이터는 리스트 형태로 유지합니다.
    """
    collated = {}
    for key in batch[0]:
        if key in ["input_ids", "attention_mask", "wrong_input_ids", "wrong_attention_mask"]:
            collated[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
        else:
            collated[key] = [item[key] for item in batch]
    return collated

def append_indices(batch, idx_list):
    """
    배치 내 각 예제에 인덱스 정보를 추가합니다.
    """
    batch["idx"] = idx_list
    return batch

# ------------------------- DataLoader 관련 함수 -------------------------

def create_data_loader(cfg, tokenizer, local_rank, num_devices):
    """
    CommonsenseQA 데이터셋을 불러오고, 전처리한 후 분산 샘플러와 DataLoader를 생성합니다.
    """
    # 데이터셋 로드 및 일부 샘플 선택
    full_dataset = load_dataset("json", data_files="CommonsenseQA/train_rand_split.jsonl")["train"]
    train_dataset = full_dataset.select(range(200))
    
    # 데이터 전처리
    train_dataset = train_dataset.map(
        lambda ex: preprocess_data(cfg, ex, tokenizer, "train"),
        batched=True
    )
    train_dataset = train_dataset.map(
        append_indices,
        with_indices=True,
        batched=True
    )
    
    dist_sampler = DistributedSampler(train_dataset, rank=local_rank, num_replicas=num_devices, shuffle=True)
    loader_kwargs = {
        'batch_size': cfg.batch_size,
        'sampler': dist_sampler,
        'collate_fn': custom_collate
    }
    cuda_opts = {
        'num_workers': 4,
        'pin_memory': True,
        'shuffle': False
    }
    loader_kwargs.update(cuda_opts)
    
    data_loader = DataLoader(train_dataset, **loader_kwargs)
    return data_loader, dist_sampler

def create_incorrect_loader(cfg, wrong_data, local_rank, num_devices):
    """
    오답 예제 리스트를 기반으로 DataLoader를 생성합니다.
    """
    wrong_dataset = Dataset.from_list(wrong_data)
    wrong_sampler = DistributedSampler(wrong_dataset, rank=local_rank, num_replicas=num_devices, shuffle=True, drop_last=True)
    
    loader_opts = {
        'batch_size': cfg.test_batch_size,
        'sampler': wrong_sampler,
        'collate_fn': custom_collate
    }
    cuda_opts = {
        'num_workers': 4,
        'pin_memory': True,
        'shuffle': False
    }
    loader_opts.update(cuda_opts)
    
    wrong_loader = DataLoader(wrong_dataset, **loader_opts)
    return wrong_loader, wrong_sampler

# ------------------------- 모델 및 FSDP 래핑 관련 함수 -------------------------

def wrap_with_fsdp(cfg, model, local_rank, offload_to_cpu=False):
    """
    주어진 모델을 FSDP로 래핑합니다.
    bf16 precision인 경우 MixedPrecision 설정을 적용합니다.
    """
    if cfg.precision == "bf16":
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        )
    else:
        mp_policy = None

    # Full state dict 설정: rank0만 CPU 오프로딩
    fsdp_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    cfg.cfg = fsdp_config

    auto_wrap = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
    
    wrapped_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        cpu_offload=offload_to_cpu,
        device_id=torch.device(local_rank)
    )
    return wrapped_model

def load_model_and_tokenizer(cfg, model_identifier, local_rank, eval_mode=False):
    """
    지정된 모델 이름으로 모델과 토크나이저를 로드한 후, FSDP로 래핑합니다.
    """
    if cfg.precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = None

    model_instance = AutoModelForCausalLM.from_pretrained(model_identifier, torch_dtype=torch.bfloat16)
    tokenizer_instance = AutoTokenizer.from_pretrained(model_identifier)
    if tokenizer_instance.pad_token is None:
        tokenizer_instance.pad_token = tokenizer_instance.eos_token
    tokenizer_instance.padding_side = "left"

    model_instance = wrap_with_fsdp(cfg, model_instance, local_rank)
    return model_instance, tokenizer_instance

# ------------------------- 분산 환경 설정 -------------------------

def init_distributed(local_rank, num_devices):
    """
    분산 환경 초기화를 위해 MASTER 주소와 포트를 지정하고 process group을 초기화합니다.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=local_rank, world_size=num_devices)

def cleanup_distributed():
    """
    분산 환경 종료 시 process group을 정리합니다.
    """
    dist.destroy_process_group()
