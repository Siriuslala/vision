"""Explicit generation settings and reproducible sampling batch boundaries."""
import argparse
import hashlib
import math


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('must be finite and greater than zero')
    return result


def probability(value: str) -> float:
    result = positive_float(value)
    if result > 1:
        raise argparse.ArgumentTypeError('must be in (0, 1]')
    return result


def min_probability(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise argparse.ArgumentTypeError('must be in [0, 1]')
    return result


def add_decoding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--do-sample', action='store_true', help='Enable multinomial sampling; default is greedy')
    parser.add_argument('--temperature', type=positive_float, default=0.7)
    parser.add_argument('--top-k', type=nonnegative_int, default=20, help='Sampling candidate limit; 0 disables it')
    parser.add_argument('--top-p', type=probability, default=0.8, help='Nucleus sampling probability mass')
    parser.add_argument('--min-p', type=min_probability, default=0.0, help='Probability threshold relative to the most likely token')
    parser.add_argument('--repetition-penalty', type=positive_float, default=1.0,
                        help='Applied in both greedy and sampling modes; 1 disables it')


def generation_kwargs(args: argparse.Namespace) -> dict:
    """Override checkpoint sampling defaults, including in greedy mode."""
    return dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
                num_beams=1, num_return_sequences=1,
                temperature=args.temperature if args.do_sample else 1.0,
                top_k=args.top_k if args.do_sample else None,
                top_p=args.top_p if args.do_sample else 1.0,
                min_p=args.min_p if args.do_sample and args.min_p > 0 else None,
                repetition_penalty=args.repetition_penalty)


def sampling_seed(seed: int, batch_index: int) -> int:
    """Give each original batch an independent, stable NumPy-compatible seed."""
    digest = hashlib.sha256(f'{seed}:{batch_index}'.encode()).digest()
    return int.from_bytes(digest[:4], 'big')


def pending_batches(rows: list[dict], done: set, id_key: str, batchsize: int,
                    do_sample: bool):
    """Replay a partially written sampling batch without regrouping its samples."""
    source = rows if do_sample else [row for row in rows if row[id_key] not in done]
    for start in range(0, len(source), batchsize):
        batch = source[start:start + batchsize]
        if any(row[id_key] not in done for row in batch):
            yield start // batchsize, batch
