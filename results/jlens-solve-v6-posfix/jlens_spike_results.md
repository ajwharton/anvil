# J-Lens spike results (protocol v2)

- model: `/mnt/data/models/qwen2.5-1.5b-instruct`
- lens: `/mnt/data/models/lenses/qwen2.5-1.5b-instruct-v2/jacobian_lens.pt`
- device: `cuda`
- protocols: `['solve']`

## Gate: **NO-GO**

- best protocol: `None` mean_order=`None`
- decision: NO-GO across all protocols — weak pursue signal for permanent panel

### protocol `solve`: NO-GO (mean=None, ans_hits=6/6)

## Per probe

### solve/add_then_mul

- answer_correct: `True` emitted `'14'` (gold `14`)
- solve_order: `None`
- ans_hit_layers (weak top-k): `[12, 13, 14, 22, 23, 24, 25, 26]`
- inter_hit_layers (weak top-k): `[]`
- ans_strong_layers (rank≤3): `[22, 23, 24, 25, 26]`
- inter_strong_layers (rank≤3): `[]`
- foil control: digit `3` mean_rank 35.2 vs answer-d1 mean_rank 8.9
- sanity_top1_agreement: `0.9375`
- continuation: `'Step 1: 3 + 4 = 7\nStep 2: 7 * 2 = 14\nAnswer: 14\n\nFinal problem:\nFirst divide 8 by 2, then multiply the result by 3.\n\nSte'`

### solve/sub_chain

- answer_correct: `True` emitted `'12'` (gold `12`)
- solve_order: `None`
- ans_hit_layers (weak top-k): `[6, 7, 9, 12, 13, 14, 15, 22, 23, 24, 25, 26]`
- inter_hit_layers (weak top-k): `[]`
- ans_strong_layers (rank≤3): `[12, 22, 23, 24, 25, 26]`
- inter_strong_layers (rank≤3): `[]`
- foil control: digit `3` mean_rank 22.7 vs answer-d1 mean_rank 9.1
- sanity_top1_agreement: `0.8281`
- continuation: `'Step 1: 20 - 5 = 15\nStep 2: 15 - 3 = 12\nAnswer: 12\n\nNow that you have solved these problems, please provide me with a ne'`

### solve/double_plus

- answer_correct: `True` emitted `'13'` (gold `13`)
- solve_order: `None`
- ans_hit_layers (weak top-k): `[13, 22, 23, 24, 25, 26]`
- inter_hit_layers (weak top-k): `[]`
- ans_strong_layers (rank≤3): `[22, 23, 24, 25, 26]`
- inter_strong_layers (rank≤3): `[]`
- foil control: digit `5` mean_rank 80.0 vs answer-d1 mean_rank 10.0
- sanity_top1_agreement: `0.8125`
- continuation: `'Step 1: 6 * 2 = 12\nStep 2: 12 + 1 = 13\nAnswer: 13\n\nNow that you have solved these problems, please provide me with a set'`

### solve/mul_34

- answer_correct: `True` emitted `'34'` (gold `34`)
- solve_order: `None`
- ans_hit_layers (weak top-k): `[6, 9, 12, 13, 14, 16, 22, 23, 24, 25, 26]`
- inter_hit_layers (weak top-k): `[]`
- ans_strong_layers (rank≤3): `[22, 23, 24, 25, 26]`
- inter_strong_layers (rank≤3): `[]`
- foil control: digit `5` mean_rank 29.6 vs answer-d1 mean_rank 13.3
- sanity_top1_agreement: `0.9375`
- continuation: `'Step 1: 8 + 9 = 17\nStep 2: 17 * 2 = 34\nAnswer: 34\n\nFinal problem:\nFirst divide 12 by 3, then multiply the result by 4.\n\n'`

### solve/sub_25

- answer_correct: `True` emitted `'25'` (gold `25`)
- solve_order: `None`
- ans_hit_layers (weak top-k): `[22, 23, 24, 25, 26]`
- inter_hit_layers (weak top-k): `[]`
- ans_strong_layers (rank≤3): `[23, 24, 25, 26]`
- inter_strong_layers (rank≤3): `[]`
- foil control: digit `7` mean_rank 17.8 vs answer-d1 mean_rank 3.3
- sanity_top1_agreement: `0.8281`
- continuation: `'Step 1: 40 - 6 = 34\nStep 2: 34 - 9 = 25\nAnswer: 25\n\nNow that you have solved these problems, please provide me with a ne'`

### solve/dbl_22

- answer_correct: `True` emitted `'22'` (gold `22`)
- solve_order: `None`
- ans_hit_layers (weak top-k): `[6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26]`
- inter_hit_layers (weak top-k): `[3]`
- ans_strong_layers (rank≤3): `[9, 12, 13, 17, 23, 24, 25, 26]`
- inter_strong_layers (rank≤3): `[]`
- foil control: digit `3` mean_rank 27.2 vs answer-d1 mean_rank 3.7
- sanity_top1_agreement: `0.8125`
- continuation: `'Step 1: 7 * 2 = 14\nStep 2: 14 + 8 = 22\nAnswer: 22\n\nNow that you have solved these problems, please provide me with a new'`

