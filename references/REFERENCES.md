# References

Since GPT2 and T5 came up only after countless attempts of different models (BERT2GPT2, BERT2BERT, other models, all with several initial checkpoints), I will leave references to articles related to all models.

- A good article explaining BERT internals (with code snippets!): https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial
- The main idea about using pre-trained checkpoints as a base for my network: https://arxiv.org/abs/1907.12461
- Converting PyTorch dataset to HuggingFace dataset: https://github.com/huggingface/datasets/issues/4983
- Seq2SeqTrainer something (but didn't work): https://pub.towardsai.net/how-to-train-a-seq2seq-summarization-model-using-bert-as-both-encoder-and-decoder-bert2bert-2a5fb36559b8
- And the notebook for the above article: https://github.com/AlaFalaki/tutorial_notebooks/blob/main/summarization/hf_BERT-BERT_training.ipynb
- A very good example of Bert2GPT2 (preparing HuggingFace dataset & how to train EncoderDecoder models): https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16
- NVIDIA, the mother of ML training beasts, tells about transformers: https://blogs.nvidia.com/blog/2022/03/25/what-is-a-transformer-model/
- A great Habr article about GPT: https://habr.com/ru/companies/ods/articles/716918/
