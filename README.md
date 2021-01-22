## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) (currently tested on version 1.6.0)
- [Transformers](http://huggingface.co/transformers/) (version 3.0.2, unlikely to work with a different version)

# I. Fusion-in-Decoder

### Download data

In what follows we explain how you can train our Fusion-in-Decoder model and download pretrained models.

### Train

[`train.py`](train.py) provides the code to train a model from scratch. An example usage of the script with some options is given below:

```shell
python train.py \
  --use_checkpoint \
  --train_data_path $tp \
  --eval_data_path $dp \
  --model_size base \
  --per_gpu_batch_size 1 \
  --n_context 100 \
  --name my_experiment \
  --checkpoint_dir checkpoint \
```  

### Test

[`test.py`](test.py) provides the script to evaluate the performance of the model. An example usage of the script is provided below.

```shell
python test.py \
  --model_path my_model_path \
  --eval_data_path my_test_data.json \
  --per_gpu_batch_size 4 \
  --n_context 100 \
  --name my_test \
  --checkpoint_dir checkpoint \
```  

### Data format

The expected data format is a list of entry examples, where each entry example is a dictionary containing
- `id`: example id, optional
- `question`: question text
- `target`: answer used for model training, if not given, the target is randomly sampled from the 'answers' list
- `answers`: list of answer text for evaluation, also used for training if target is not given
- `ctxs`: a list of passages where each item is a dictionary containing
        - `title`: article title
        - `text`: passage text

Entry example:
```
{
  'id': '0',
  'question': 'What element did Marie Curie name after her native land?',
  'target': 'Polonium',
  'answers': ['Polonium', 'Po (chemical element)', 'Po'],
  'ctxs': [
            {
                "title": "Marie Curie",
                "text": "them on visits to Poland. She named the first chemical element that she discovered in 1898 \"polonium\", after her native country. Marie Curie died in 1934, aged 66, at a sanatorium in Sancellemoz (Haute-Savoie), France, of aplastic anemia from exposure to radiation in the course of her scientific research and in the course of her radiological work at field hospitals during World War I. Maria Sk\u0142odowska was born in Warsaw, in Congress Poland in the Russian Empire, on 7 November 1867, the fifth and youngest child of well-known teachers Bronis\u0142awa, \"n\u00e9e\" Boguska, and W\u0142adys\u0142aw Sk\u0142odowski. The elder siblings of Maria"
            },
            {
                "title": "Marie Curie",
                "text": "was present in such minute quantities that they would eventually have to process tons of the ore. In July 1898, Curie and her husband published a joint paper announcing the existence of an element which they named \"polonium\", in honour of her native Poland, which would for another twenty years remain partitioned among three empires (Russian, Austrian, and Prussian). On 26 December 1898, the Curies announced the existence of a second element, which they named \"radium\", from the Latin word for \"ray\". In the course of their research, they also coined the word \"radioactivity\". To prove their discoveries beyond any"
            }
          ]
}
```

# II. Using cross-attention scores


## References

[1] G. Izacard, E. Grave [*Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering*](https://arxiv.org/abs/2007.01282)

```
@misc{izacard2020leveraging,
      title={Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering}, 
      author={Gautier Izacard and Edouard Grave},
      year={2020},
      eprint={2007.01282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Distilling Knowledge from Reader to Retriever for Question Answering

[2] G. Izacard, E. Grave [*Distilling Knowledge from Reader to Retriever for Question Answering*](https://arxiv.org/abs/2012.04584)

```
@misc{izacard2020distilling,
      title={Distilling Knowledge from Reader to Retriever for Question Answering}, 
      author={Gautier Izacard and Edouard Grave},
      year={2020},
      eprint={2012.04584},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[3] G. Izacard, F. Petroni, L. Hosseini, N. De Cao, S. Riedel, E. Grave [*A Memory Efficient Baseline for Open Domain Question Answering*](https://arxiv.org/abs/2012.15156)

```
@misc{izacard2020memory,
      title={A Memory Efficient Baseline for Open Domain Question Answering}, 
      author={Gautier Izacard and Fabio Petroni and Lucas Hosseini and Nicola De Cao and Sebastian Riedel and Edouard Grave},
      year={2020},
      eprint={2012.15156},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

See the [LICENSE](LICENSE) file for more details.



