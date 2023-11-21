<img src="./rt1.png" width="450px"></img>

## Robotic Transformer - Pytorch

Implementation of <a href="https://ai.googleblog.com/2022/12/rt-1-robotics-transformer-for-real.html">RT1 (Robotic Transformer)</a>, from the Robotics at Google team, in Pytorch

## Instructions

```bash
git clone https://github.com/Rohan138/rt1-pytorch.git
pip install -e .

# Make sure to run this! See https://github.com/pytorch/pytorch/pull/114083/files
bash fix_torch.sh

# See main.py for additional arguments
python main.py --train-split "train[:1000]" --eval-split "train[:1000]" --train-batch-size 8 --eval-batch-size 8 --eval-freq 100 --checkpoint-freq 1000
```

### Changelog
- 11/10/2023: Initial commit! Wrote FiLM-EfficientNet and tests
- 11/11/2023: Wrote tokenizers and initial RT1 skeleton; separate `tests` folder 
download and experiment with datasets
- 11/14/2023: Separate `RT1Model` and `RT1Policy`; fix einops
- 11/16/2023: Add `data.py`; add back `USE` embeddings
- 11/19/2023: Lots of cleanup; move trajectory logic to `data.py`; found 
PyTorch [bug](https://github.com/pytorch/pytorch/pull/114083/files), added tests
for model, policy, and loss; fix action tokenizer
- 11/20/2023: Updates to `data.py`; add `device=cuda` support; add `fix_torch.sh`; 
finalize `main.py`; add simple evaluation and checkpointing logic


### TODO
- Add smaller `d4rl` benchmarks
- Add off-policy evaluation to evaluate on test dataset
- Implement DDP; make sure we check loss reduction stays the same
- Try onehot encoding discrete actions instead of passing the raw action as a token
- Optimize so we don't run tokenizer 6x per image; see `efficient-encode` branch
- Switch to predicting last token only; see `lasttoken` branch
- Try [CoW-MOO](https://robot-moo.github.io/)
- Try ViT instead as in the `lucidrains` implementation
- Hyperparameter tuning
- Tests! All the tests! Unit tests, learning tests, testing all the way!
- Train and evaluate on real KUKA robot
- Setup lint and CI pipeline; currently running `black` and `isort`

## Acknowledgements

Special thanks to [Raghava Uppuluri](https://github.com/raghavauppuluri13) 
for all the ideas, late-night deubgging discussions, compute, hardware, and support!

Initial implementation borrowed from [lucidrains/robotic-transformer-pytorch](https://github.com/lucidrains/robotic-transformer-pytorch)

[Datasets](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit#gid=0)

```bibtex
@inproceedings{rt12022arxiv,
    title    = {RT-1: Robotics Transformer for Real-World Control at Scale},
    author   = {Anthony Brohan and Noah Brown and Justice Carbajal and  Yevgen Chebotar and Joseph Dabis and Chelsea Finn and Keerthana Gopalakrishnan and Karol Hausman and Alex Herzog and Jasmine Hsu and Julian Ibarz and Brian Ichter and Alex Irpan and Tomas Jackson and  Sally Jesmonth and Nikhil Joshi and Ryan Julian and Dmitry Kalashnikov and Yuheng Kuang and Isabel Leal and Kuang-Huei Lee and  Sergey Levine and Yao Lu and Utsav Malla and Deeksha Manjunath and  Igor Mordatch and Ofir Nachum and Carolina Parada and Jodilyn Peralta and Emily Perez and Karl Pertsch and Jornell Quiambao and  Kanishka Rao and Michael Ryoo and Grecia Salazar and Pannag Sanketi and Kevin Sayed and Jaspiar Singh and Sumedh Sontakke and Austin Stone and Clayton Tan and Huong Tran and Vincent Vanhoucke and Steve Vega and Quan Vuong and Fei Xia and Ted Xiao and Peng Xu and Sichun Xu and Tianhe Yu and Brianna Zitkovich},
    booktitle = {arXiv preprint arXiv:2204.01691},
    year      = {2022}
}
```
