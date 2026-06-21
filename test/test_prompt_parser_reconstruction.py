import unittest

import torch

from modules import prompt_parser


def scheduled_batch(*schedules):
    return prompt_parser.ScheduledConditioningBatch([
        prompt_parser.ScheduledConditioning(schedule) for schedule in schedules
    ])


def legacy_target_index(cond_schedule, current_step):
    for index, entry in enumerate(cond_schedule):
        if current_step <= entry.end_at_step:
            return index
    return 0


def legacy_reconstruct_cond_batch(c, current_step):
    param = c[0][0].cond
    if isinstance(param, dict):
        res = {k: torch.zeros((len(c),) + value.shape, device=value.device, dtype=value.dtype) for k, value in param.items()}
        res = prompt_parser.DictWithShape(res, (len(c),) + param["crossattn"].shape)
    else:
        res = torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype)

    for i, cond_schedule in enumerate(c):
        cond = cond_schedule[legacy_target_index(cond_schedule, current_step)].cond
        if isinstance(cond, dict):
            for k, value in cond.items():
                res[k][i] = value
        else:
            res[i] = cond

    return res


def legacy_stack_conds(tensors):
    tensors = list(tensors)
    token_count = max(x.shape[0] for x in tensors)
    for i, tensor in enumerate(tensors):
        if tensor.shape[0] != token_count:
            last_vector = tensor[-1:]
            tensors[i] = torch.vstack([tensor, last_vector.repeat([token_count - tensor.shape[0], 1])])

    return torch.stack(tensors)


class PromptParserReconstructionTest(unittest.TestCase):
    def test_reconstruct_cond_batch_reuses_tensor_buffer(self):
        first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        second = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = scheduled_batch(
            [prompt_parser.ScheduledPromptConditioning(10, first)],
            [prompt_parser.ScheduledPromptConditioning(10, second)],
        )

        result = prompt_parser.reconstruct_cond_batch(c, 0)
        again = prompt_parser.reconstruct_cond_batch(c, 1)

        self.assertIs(result, again)
        self.assertTrue(torch.equal(legacy_reconstruct_cond_batch(c, 1), again))

    def test_reconstruct_cond_batch_matches_legacy_when_schedule_advances(self):
        c = scheduled_batch(
            [
                prompt_parser.ScheduledPromptConditioning(2, torch.tensor([[1.0, 2.0]])),
                prompt_parser.ScheduledPromptConditioning(5, torch.tensor([[3.0, 4.0]])),
            ],
            [
                prompt_parser.ScheduledPromptConditioning(2, torch.tensor([[5.0, 6.0]])),
                prompt_parser.ScheduledPromptConditioning(5, torch.tensor([[7.0, 8.0]])),
            ],
        )

        self.assertTrue(torch.equal(legacy_reconstruct_cond_batch(c, 1), prompt_parser.reconstruct_cond_batch(c, 1)))
        reused = prompt_parser.reconstruct_cond_batch(c, 3)

        self.assertTrue(torch.equal(legacy_reconstruct_cond_batch(c, 3), reused))

    def test_reconstruct_cond_batch_reuses_dict_buffers(self):
        c = scheduled_batch(
            [prompt_parser.ScheduledPromptConditioning(10, {
                "crossattn": torch.tensor([[1.0, 2.0]]),
                "vector": torch.tensor([3.0, 4.0]),
            })],
            [prompt_parser.ScheduledPromptConditioning(10, {
                "crossattn": torch.tensor([[5.0, 6.0]]),
                "vector": torch.tensor([7.0, 8.0]),
            })],
        )

        result = prompt_parser.reconstruct_cond_batch(c, 0)
        crossattn = result["crossattn"]
        vector = result["vector"]
        again = prompt_parser.reconstruct_cond_batch(c, 1)

        self.assertIs(result, again)
        self.assertIs(crossattn, again["crossattn"])
        self.assertIs(vector, again["vector"])
        expected = legacy_reconstruct_cond_batch(c, 1)
        self.assertTrue(torch.equal(expected["crossattn"], again["crossattn"]))
        self.assertTrue(torch.equal(expected["vector"], again["vector"]))

    def test_reconstruct_multicond_batch_reuses_buffer_and_pads_shorter_conds(self):
        schedule_a = prompt_parser.ScheduledConditioning([
            prompt_parser.ScheduledPromptConditioning(10, torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ])
        schedule_b = prompt_parser.ScheduledConditioning([
            prompt_parser.ScheduledPromptConditioning(10, torch.tensor([[5.0, 6.0]])),
        ])
        conditioning = prompt_parser.MulticondLearnedConditioning(
            shape=(2,),
            batch=[
                [prompt_parser.ComposableScheduledPromptConditioning(schedule_a, 0.7)],
                [prompt_parser.ComposableScheduledPromptConditioning(schedule_b, 1.3)],
            ],
        )

        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(conditioning, 0)
        again_conds_list, again = prompt_parser.reconstruct_multicond_batch(conditioning, 1)

        expected = legacy_stack_conds([schedule_a[0].cond, schedule_b[0].cond])
        self.assertEqual([[(0, 0.7)], [(1, 1.3)]], conds_list)
        self.assertEqual(conds_list, again_conds_list)
        self.assertIs(tensor, again)
        self.assertTrue(torch.equal(expected, again))


if __name__ == "__main__":
    unittest.main()
