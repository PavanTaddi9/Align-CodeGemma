import logging
import os
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser

from execserver.code_exec_reqs import exec_test_batched
import utils
from prompt_template import format_instruction

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

@dataclass
class ScriptArguments:
    train_path: str = "datas/train_meta.json"
    test_path: str = "datas/test_meta.json"
    tokenizer_name: Optional[str] = None

# Unified logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def process_completions(completions: List[List[dict]]) -> List[str]:
    """Extract and clean Python code from completions."""
    processed = []
    for comp_group in completions:
        if not comp_group or "content" not in comp_group[0]:
            processed.append("")
            continue
            
        content = comp_group[0]["content"]
        code_blocks = utils.find_code_blocks(content, tag="python")
        processed.append("\n\n".join(code_blocks).replace('\\"""', '"""') if code_blocks else "")
    return processed

def run_tests_and_reward(
    prompts: List[str],
    completions: List[List[dict]],
    timeout: int = 60
) -> List[int]:
    return exec_test_batched(
        "http://localhost:8000",
        process_completions(completions),
        timeout=timeout
    )

def reward_based_on_jax_usage(
    prompts: List[str],
    completions: List[List[dict]]
) -> List[float]:
    return [utils.count_jax_usage(code) for code in process_completions(completions)]

def load_formatted_dataset(path: str) -> Dataset:
    """Load and format dataset from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return Dataset.from_list([
            {"prompt": format_instruction(item["instruction"])}
            for item in json.load(f)
        ])

def grpo_trainer_setup(args: tuple) -> GRPOTrainer:
    """Configure and return GRPO trainer."""
    model_args, script_args, training_args = args
    
    # Tokenizer initialization
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name or model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    return GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[run_tests_and_reward, reward_based_on_jax_usage],
        args=training_args,
        train_dataset=load_formatted_dataset(script_args.train_path),
        peft_config=get_peft_config(model_args),
    )

def main():
    # Argument parsing
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    args = parser.parse_args_and_config()
    
    # Training setup
    trainer = grpo_trainer_setup(args)
    training_args = args[2]
    
    # Checkpoint handling
    checkpoint = get_last_checkpoint(training_args.output_dir) \
        if os.path.isdir(training_args.output_dir) else None
    
    logger.info(f"Training starting at {datetime.now():%Y-%m-%d %H:%M:%S}")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Results processing
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    
    # Model saving
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    trainer.tokenizer.save_pretrained(training_args.output_dir)
    
    if training_args.push_to_hub:
        logger.info("Uploading to Hub...")
        trainer.push_to_hub()
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()