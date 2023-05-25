import json
import os
import uuid
import time
import torch

import logging
import time
from accelerate.logging import get_logger

logger = get_logger(__name__)
CHECKPOINT_META_NAME = "checkpoint.meta"

def init_logs(log_dir, rank):
    cur_date  = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    logging.basicConfig(
        filename=os.path.join(log_dir, f"{cur_date}_train_{rank}.log"),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )



def load_latest_ckpt_meta(ckpt: str):
    if len(ckpt) == 0:
        return

    meta = {}
    try:
        with open(ckpt, "r") as input:
            data = input.read()
            if len(data) > 0:
                meta = json.loads(data)
    except Exception as e:
        print("load_latest_ckpt_meta catch exception:", e)

    return meta


def save_latest_ckpt_meta(ckpt: str, meta: dict):
    if len(meta) == 0:
        return
    
    try:
        file_prefix = os.path.splitext(ckpt)[0]
        data = json.dumps(meta)
        uid = uuid.uuid4().__str__()
        tmp_file = file_prefix + "_" + uid + ".tmp"
        with open(tmp_file, 'w') as output:
            output.write(json.dumps(meta))

        os.rename(tmp_file, ckpt)
    except Exception as e:
        logger.info("save_latest_ckpt_meta catch exception:", e)

    return



def save_ckpt(prefix:str, model, optimizer, rank, epoch, step):
    try:
        ckpts = os.join(prefix, "checkpoints")
        os.mkdir(ckpts)
    except Exception as e:
        pass

    cur_date = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    meta = {}
    
    state = f'{cur_date}_rank_{rank}_epoch_{epoch}_step{step}'
    model_file = os.path.join(ckpts, f'{state}.pt')
    meta.update("model", model_file)
    torch.save({model, model_file})

    optim_file = os.path.join(ckpts, f'{state}.opt')
    torch.save(optimizer, optim_file)
    meta.update({"optimizer", model_file})

    meta.update({"epoch", epoch})
    meta.update({"step", step})
    meta.update({"rank", rank})

    save_latest_ckpt_meta(os.path.join(prefix, CHECKPOINT_META_NAME), meta)


def load_ckpt(prefix: str):
    meta = load_latest_ckpt_meta(os.path.join(prefix, CHECKPOINT_META_NAME))
    if not meta:
        return None, None, None, None
    
    try:
        if "model" in meta:
            model = torch.load(meta["model"])
        if "optim" in meta:
            opt = torch.load(meta["optimizer"])
    except Exception as e:
        raise(f"model load exception: {e}")
        
    return model, opt, meta["epoch"], meta["step"]



def timecost_wrapper(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_cost = end_time - start_time

        cost = {func.__name__ : time_cost}
        logger.info(f"time cost statistics: {cost}")
        return result
    
    return wrapper



class TimeTicker:
    def __init__(self, name, switch=True):
        self.start_time = None
        self.end_time = None
        self.time_cost = None
        self.name  = name
        self.switch = switch
        self.reporter = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.time_cost = self.end_time - self.start_time

        if self.switch:
            logger.info(f"time cost op:{self.name}, cost:{self.time_cost}")