import argparse


class CustomArgumentParser:
    @staticmethod
    def parse_opt():
        parser = argparse.ArgumentParser()

        # Load LLM
        parser.add_argument('-oa', '--openai', action='store_true', help='Load OpenAI LLM')
        parser.add_argument('-ot', '--openthaigpt', action='store_true', help='Load OpenThaiGPT LLM')
        parser.add_argument('-wc', '--wangchanglm', action='store_true', help='Load WangChanGLM LLM')
        parser.add_argument('-ga', '--gpt4all', action='store_true', help='Load GPT4All LLM')

        # Set Chunk Embedding
        parser.add_argument('-cs', '--chunksize', type=int, default=1000, help='Set Chunk Size')
        parser.add_argument('-co', '--chunkoverlap', type=int, default=200, help='Set Chunk Overlap')

        # Source Documents Directory
        parser.add_argument('-s', '--source', action='store_true', help="Set Path to Documents Directory")

        # parser.set_defaults(openai=True)
        opt = parser.parse_args()
        return opt


    @staticmethod
    def qlora_parse_opt():
        parser = argparse.ArgumentParser()

        # Dataset Directory and Resume from checkpoint
        parser.add_argument('--dataset_dir', type=str, default="", help="Set path to dataset directory")
        parser.add_argument('--resume_from_checkpoint', type=str, default="", help="Resume from training checkpoint or final adapter path")

        # Load LLM
        parser.add_argument('--openthaigpt', action='store_true', help='Load OpenThaiGPT LLM')
        parser.add_argument('--wangchanglm', action='store_true', help='Load WangChanGLM LLM')

        # Lora Config
        parser.add_argument('--lora_r', type=int, default=16, help='Set Lora Rank')
        parser.add_argument('--lora_alpha', type=int, default=32, help='Set Lora Alpha')
        parser.add_argument('--lora_dropout', type=float, default=0.05, help='Set Lora Dropout')
        parser.add_argument('--bias', type=str, default="none", help='Set Lora Bias')
        
        # Training Arguments
        parser.add_argument('--output_dir', type=str, default="experiments", help='The output directory where the model predictions and checkpoints will be written')
        parser.add_argument('--per_device_train_batch_size', type=int, default=4, help='The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training')
        parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of updates steps to accumulate the gradients for, before performing a backward/update pass')
        parser.add_argument('--optim', type=str, default="paged_adamw_32bit", help='The optimizer to use: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or adafactor.')
        parser.add_argument('--logging_steps', type=int, default=10, help='Number of update steps between two logs. Should be in range `[0, 1)`')
        parser.add_argument('--learning_rate', type=float, default=1e-4, help='The initial learning rate for optimizer')
        parser.add_argument('--max_grad_norm', type=float, default=0.3, help='Maximum gradient norm (for gradient clipping)')
        parser.add_argument('--num_train_epochs', type=int, default=2, help='Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training)')
        parser.add_argument('--evaluation_strategy', type=str, default="steps", help='The evaluation strategy to adopt during training. Possible values are `no`, `steps`, `epoch`')
        parser.add_argument('--eval_steps', type=float, default=0.2, help='Number of update steps between two evaluations. Should be in range `[0, 1)`')
        parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Ratio of total training steps used for a linear warmup from 0 to `learning_rate`')
        parser.add_argument('--save_strategy', type=str, default="epoch", help='The checkpoint save strategy to adopt during training. Possible values are `no`, `steps`, `epoch`')
        parser.add_argument('--lr_scheduler_type', type=str, default="cosine", help='The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values')
        parser.add_argument('--fp16', type=bool, default=True, help='Whether to use fp16 (mixed) precision instead of 32-bit')
        parser.add_argument('--group_by_length', type=bool, default=True, help='Whether or not to group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient). Only useful if applying dynamic padding.')
        parser.add_argument('--save_safetensors', type=bool, default=True, help='Use saving and loading for state dicts instead of default `torch.load` and `torch.save`')

        # parser.set_defaults(wangchanglm=True)
        opt = parser.parse_args()
        return opt