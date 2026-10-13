"""Check decoding validation, real CPU generation, and sampling replay."""
import argparse
import contextlib
import io
import unittest

from eva.decoding import add_decoding_arguments, generation_kwargs, pending_batches, sampling_seed


class DecodingTests(unittest.TestCase):
    def args(self, *flags):
        parser = argparse.ArgumentParser()
        add_decoding_arguments(parser)
        args = parser.parse_args(flags)
        args.max_new_tokens = 5
        return args

    def test_invalid_settings(self):
        invalid = {'--temperature': ['0', '-1', 'nan', 'inf'],
                   '--top-k': ['-1', '1.5'], '--top-p': ['0', '1.1', 'nan'],
                   '--min-p': ['-0.1', '1.1', 'inf'], '--repetition-penalty': ['0', 'nan']}
        for flag, values in invalid.items():
            for value in values:
                with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        self.args(flag, value)

    def test_cpu_generation_and_sampling_replay(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, StoppingCriteriaList, set_seed
        from eva.progress import GenerationProgress

        set_seed(0)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=16,
                                         n_embd=16, n_layer=1, n_head=2,
                                         bos_token_id=1, eos_token_id=None, pad_token_id=0)).eval()
        # Simulate a checkpoint with sampling defaults that must be overridden.
        model.generation_config.do_sample = True
        model.generation_config.top_k = 7
        model.generation_config.top_p = 0.2
        rows = [{'id': i} for i in range(5)]

        def generate(done, args):
            outputs = {}
            with torch.inference_mode():
                for index, batch in pending_batches(rows, done, 'id', 2, args.do_sample):
                    set_seed(sampling_seed(42, index))
                    inputs = torch.tensor([[1, r['id'] + 2] for r in batch])
                    tokens = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                            **generation_kwargs(args))
                    outputs.update({r['id']: seq.tolist() for r, seq in zip(batch, tokens)
                                    if r['id'] not in done})
            return outputs

        for args in [self.args(), self.args('--do-sample'),
                     self.args('--do-sample', '--top-k', '0', '--top-p', '1', '--min-p', '0.1')]:
            full = generate(set(), args)
            resumed = generate({0, 1, 2}, args)
            self.assertEqual(resumed, {key: value for key, value in full.items() if key > 2})
        inputs = torch.tensor([[1, 2]])
        with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
            expected = model.generate(inputs, **generation_kwargs(self.args()))
            with GenerationProgress(2, 1) as progress:
                actual = model.generate(inputs, **generation_kwargs(self.args()),
                                        stopping_criteria=StoppingCriteriaList([progress]))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(progress.steps, 5)
        self.assertIsNotNone(progress.first_token_seconds)
        self.assertFalse(generation_kwargs(self.args())['do_sample'])
        self.assertIsNone(generation_kwargs(self.args())['top_k'])
        batches = list(pending_batches(rows, {0, 1, 2}, 'id', 2, True))
        self.assertEqual([(i, [r['id'] for r in batch]) for i, batch in batches], [(1, [2, 3]), (2, [4])])
