import os
import argparse
import math
from config.base_config import Config
from modules.basic_utils import mkdirp, deletedir


class AllConfig(Config):
    def __init__(self):
        super().__init__()

    def parse_args(self):
        description = 'Text-to-Video Retrieval'
        parser = argparse.ArgumentParser(description=description)
        
        # data parameters
        parser.add_argument('--dataset_name', type=str, default='MSRVTT', help="Dataset name")
        parser.add_argument('--videos_dir', type=str, default='data/MSRVTT/vids', help="Location of videos")
        parser.add_argument('--msrvtt_train_file', type=str, default='9k')
        parser.add_argument('--num_frames', type=int, default=12)
        parser.add_argument('--video_sample_type', default='uniform', help="'rand'/'uniform'")
        parser.add_argument('--input_res', type=int, default=224)

        # experiment parameters
        parser.add_argument('--exp_name', type=str, required=True, help="Name of the current experiment")
        parser.add_argument('--output_dir', type=str, default='./outputs')
        parser.add_argument('--save_every', type=int, default=1, help="Save model every n epochs")
        parser.add_argument('--log_step', type=int, default=10, help="Print training log every n steps")
        parser.add_argument('--evals_per_epoch', type=int, default=10, help="Number of times to evaluate per epoch")
        parser.add_argument('--eval_batch_size', type=int, default=None, help='Validation batch size; defaults to at most 4 with chunked training')
        parser.add_argument('--load_epoch', type=int, help="Epoch to load from exp_name, or -1 to load model_best.pth")
        parser.add_argument('--eval_window_size', type=int, default=5, help="Size of window to average metrics")
        parser.add_argument('--metric', type=str, default='t2v', help="'t2v'/'v2t'")

        # model parameters
        parser.add_argument('--huggingface', action='store_true', default=False)
        parser.add_argument('--clip_model_path', help='Local Hugging Face CLIP model and tokenizer directory')
        parser.add_argument('--arch', type=str, default='clip_transformer')
        parser.add_argument('--clip_arch', type=str, default='ViT-B/32', help="CLIP arch. only when not using huggingface")
        parser.add_argument('--embed_dim', type=int, default=512, help="Dimensionality of the model embedding")

        # training parameters
        parser.add_argument('--loss', choices=('clip', 'supcon'), default='clip')
        parser.add_argument('--clip_lr', type=float, default=1e-6, help='Learning rate used for CLIP params')
        parser.add_argument('--noclip_lr', type=float, default=1e-5, help='Learning rate used for new params')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--grad_cache_chunk_size', type=int, default=0, help='0: off; otherwise effective batch is encoded in chunks of this size')
        parser.add_argument('--accumulation_chunk_size', type=int, default=0, help='0: off; otherwise compute an independent contrastive loss per chunk and update once per effective batch')
        parser.add_argument('--grad_cache_check_recompute', action='store_true', help='Verify deterministic feature recomputation every step (slower)')
        parser.add_argument('--num_epochs', type=int, default=5)
        parser.add_argument('--weight_decay', type=float, default=0.2, help='Weight decay')
        parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion for learning rate schedule')
        parser.add_argument('--max_train_seconds', type=float, default=0, help='Stop training after this many seconds, or 0 for no limit')
        parser.add_argument('--max_train_steps', type=int, default=0, help='Stop after this many optimizer steps, or 0 for no limit')
        parser.add_argument('--profile_training', action='store_true', help='Record step time, loss and CUDA memory')
        parser.add_argument('--resume', type=str, help='Training checkpoint path to resume, including a partial epoch')

        # frame pooling parameters
        parser.add_argument('--pooling_type', type=str)
        parser.add_argument('--k', type=int, default=-1, help='K value for topk pooling')
        parser.add_argument('--attention_temperature', type=float, default=0.01, help='Temperature for softmax (used in attention pooling only)')
        parser.add_argument('--num_mha_heads', type=int, default=1, help='Number of parallel heads in multi-headed attention')
        parser.add_argument('--transformer_dropout', type=float, default=0.3, help='Dropout prob. in the transformer pooling')

        # system parameters
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--seed', type=int, default=24, help='Random seed')
        parser.add_argument('--no_tensorboard', action='store_true', default=False)
        parser.add_argument('--tb_log_dir', type=str, default='logs')

        args = parser.parse_args()
        if args.batch_size <= 0:
            parser.error('--batch_size must be positive')
        if not 0 <= args.grad_cache_chunk_size <= args.batch_size:
            parser.error('--grad_cache_chunk_size must be 0 (off) or in [1, batch_size]')
        if not 0 <= args.accumulation_chunk_size <= args.batch_size:
            parser.error('--accumulation_chunk_size must be 0 (off) or in [1, batch_size]')
        if args.grad_cache_chunk_size and args.accumulation_chunk_size:
            parser.error('GradCache and independent-loss accumulation are mutually exclusive')
        if args.grad_cache_check_recompute and not args.grad_cache_chunk_size:
            parser.error('--grad_cache_check_recompute requires GradCache')
        if args.clip_model_path:
            if not args.huggingface:
                parser.error('--clip_model_path requires --huggingface')
            args.clip_model_path = os.path.abspath(os.path.expanduser(args.clip_model_path))
            if not os.path.isdir(args.clip_model_path):
                parser.error(f'CLIP model directory not found: {args.clip_model_path}')
        if args.grad_cache_chunk_size and (not args.huggingface or args.arch != 'clip_transformer'):
            parser.error('GradCache requires --huggingface --arch=clip_transformer')
        if args.eval_batch_size is None:
            chunked = args.grad_cache_chunk_size or args.accumulation_chunk_size
            args.eval_batch_size = min(args.batch_size, 4) if chunked else args.batch_size
        if args.eval_batch_size <= 0:
            parser.error('--eval_batch_size must be positive')
        if args.max_train_seconds < 0 or not math.isfinite(args.max_train_seconds) or args.max_train_steps < 0:
            parser.error('Training limits must be finite and nonnegative')

        args.model_path = os.path.join(args.output_dir, args.exp_name)
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.exp_name)

        mkdirp(args.model_path)
        if not args.resume:
            deletedir(args.tb_log_dir)
        mkdirp(args.tb_log_dir)

        return args
