# J-Lens spike results (protocol v2)

- model: `/mnt/data/models/qwen2.5-7b-instruct`
- lens: `/mnt/data/models/lenses/qwen2.5-7b-instruct-v0/jacobian_lens.pt`
- device: `cuda`
- protocols: `['solve']`

## Gate: **NO-GO**

- best protocol: `solve` mean_order=`0.5`
- decision: NO-GO across all protocols — weak pursue signal for permanent panel

### protocol `solve`: NO-GO (mean=0.5, ans_hits=6/6)

## Per probe

### solve/add_then_mul

- answer_correct: `True` emitted `'14'` (gold `14`)
- solve_order: `1.0`
- ans_hit_layers (weak top-k): `[23, 24, 25, 26]`
- inter_hit_layers (weak top-k): `[10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26]`
- ans_strong_layers (rank≤3): `[24, 25, 26]`
- inter_strong_layers (rank≤3): `[13, 14, 16, 22, 23, 24, 25, 26]`
- foil control: digit `3` mean_rank 9956.7 vs answer-d1 mean_rank 10401.6
- sanity_top1_agreement: `1.0`
- continuation: `'Step 1: 3 + 4 = 7\nStep 2: 7 * 2 = 14\nAnswer: 14\n\nProblem: Begin with 8, subtract 3, then add 6.\nStep 1: 8 - 3 = 5\nStep '`

### solve/sub_chain

- answer_correct: `True` emitted `'12'` (gold `12`)
- solve_order: `1.0`
- ans_hit_layers (weak top-k): `[8, 9, 10, 11, 12, 13, 24, 25, 26]`
- inter_hit_layers (weak top-k): `[12, 13, 14, 16, 24, 25, 26]`
- ans_strong_layers (rank≤3): `[25, 26]`
- inter_strong_layers (rank≤3): `[25, 26]`
- foil control: digit `3` mean_rank 3875.9 vs answer-d1 mean_rank 7327.3
- sanity_top1_agreement: `0.9844`
- continuation: `'Step 1: 20 - 5 = 15\nStep 2: 15 - 3 = 12\nAnswer: 12\n\nProblem: Start with 15, subtract 7, then subtract 2.\nStep 1: 15 - 7 '`

### solve/double_plus

- answer_correct: `True` emitted `'13'` (gold `13`)
- solve_order: `0.0`
- ans_hit_layers (weak top-k): `[24, 25, 26]`
- inter_hit_layers (weak top-k): `[6, 8, 9, 10, 11, 13, 25, 26]`
- ans_strong_layers (rank≤3): `[24, 25, 26]`
- inter_strong_layers (rank≤3): `[25, 26]`
- foil control: digit `5` mean_rank 9711.3 vs answer-d1 mean_rank 9181.3
- sanity_top1_agreement: `0.9688`
- continuation: `'Step 1: 6 * 2 = 12\nStep 2: 12 + 1 = 13\nAnswer: 13\n\nProblem: Subtract 3 from 8, then triple the difference.\nStep 1: 8 - 3'`

### solve/mul_34

- answer_correct: `True` emitted `'34'` (gold `34`)
- solve_order: `0.0`
- ans_hit_layers (weak top-k): `[13, 14, 23, 24, 25, 26]`
- inter_hit_layers (weak top-k): `[13, 24, 25, 26]`
- ans_strong_layers (rank≤3): `[24, 25, 26]`
- inter_strong_layers (rank≤3): `[25, 26]`
- foil control: digit `5` mean_rank 6881.9 vs answer-d1 mean_rank 4563.8
- sanity_top1_agreement: `0.9688`
- continuation: `'Step 1: 8 + 9 = 17\nStep 2: 17 * 2 = 34\nAnswer: 34\n\nProblem: Begin with 15, subtract 7, then divide the result by 3.\nStep'`

### solve/sub_25

- answer_correct: `True` emitted `'25'` (gold `25`)
- solve_order: `1.0`
- ans_hit_layers (weak top-k): `[16, 23, 24, 25, 26]`
- inter_hit_layers (weak top-k): `[13, 14, 16, 23, 24, 25, 26]`
- ans_strong_layers (rank≤3): `[23, 24, 25, 26]`
- inter_strong_layers (rank≤3): `[16, 23, 24, 25, 26]`
- foil control: digit `7` mean_rank 1501.1 vs answer-d1 mean_rank 3512.8
- sanity_top1_agreement: `0.9844`
- continuation: `'Step 1: 40 - 6 = 34\nStep 2: 34 - 9 = 25\nAnswer: 25\n\nProblem: Start with 35, subtract 8, then subtract 3.\nStep 1: 35 - 8 '`

### solve/dbl_22

- answer_correct: `True` emitted `'22'` (gold `22`)
- solve_order: `0.0`
- ans_hit_layers (weak top-k): `[6, 8, 9, 10, 11, 16, 23, 24, 25, 26]`
- inter_hit_layers (weak top-k): `[11, 12, 25, 26]`
- ans_strong_layers (rank≤3): `[8, 23, 24, 25, 26]`
- inter_strong_layers (rank≤3): `[25, 26]`
- foil control: digit `3` mean_rank 2184.8 vs answer-d1 mean_rank 1461.5
- sanity_top1_agreement: `0.9531`
- continuation: `'Step 1: 7 * 2 = 14\nStep 2: 14 + 8 = 22\nAnswer: 22\n\nProblem: Subtract 9 from 15, then divide the result by 3.\nStep 1: 15 '`

