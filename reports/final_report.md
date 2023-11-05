# The final report

## Solution

I ended up with two trained models: GPT2 and T5. I could not decide which one to pick as the final one, so I included notebooks and checkpoints for both.

The Python scripts in `./src` use GPT2, because it is more lightweight.

GPT2 is a decoder-only model, so during training it only accepts inputs - no expected outputs. That's the main power of GPT model family: it can consume enormous amounts of raw data, without data labelling.

But since this assignment actually requires to use expected outputs, I came up with a clever trick: I format the text in the following way:

```
User: <original text>
Assistant: <detoxified text>
```

Since the main purpose of GPT is to predict the next token, naturally, when it receives a `User: <text>` input, it goes on with `Assistant: <detoxified text>`.

One epoch of training took around 90-100 minutes. I did one epoch, then tested the output, it was OK, then I started another two epochs.


## Some words on T5

T5, contrary to GPT, is traditional encoder-decoder model, which requires it to get both input and expected output, which fits the provided dataset (and task) just right without any tricks.

I suppose T5 is slower because model creators had to choose bigger weight count, given that the original intent is to make model multi-task with large amount of different tasks.

One epoch of training took 7-8 hours. I only did one epoch, because results were good enough and I was tired of wasting so much time when GPT took only the fraction of this.


## Hardware used

Work was done using a huge variety of software. Some part was done locally on my laptop with Linux (with no acceleration, CPU-only), some on M1 MacBook (with MPS device for training/inference acceleration), some on desktop PC with RTX2060, some on Kaggle with 2 x Tesla T4, some on Google Colab with 1 x Tesla T4, and the final models were trained on Selectel VDS (Virtual Dedicated Server) with Tesla A4000.

When I trained GPT2, I fit VRAM PERFECTLY: I had 31MB free out of 16GB :) It would be a shame if some random allocation would end up failing the training process...

I tested inference on CPU and it works OK (not very fast, but still, I could manage to compute some metrics).
