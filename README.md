# Fusion-in-Decoder

This branch contains code snapshot of the FiD repo from Nov 2020.
No support will be provided for this code branch.

## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) (currently tested on version 1.6.0)
- [Transformers](http://huggingface.co/transformers/) (version 3.0.2, unlikely to work with a different version)

### Download data

### Train

[`train.py`](train.py) provides the code for training a model from scratch. An example usage of the script with some options is given below:

```shell
python train.py \
  --use_checkpointing \
  --train_data_path $tp \
  --dev_data_path $dp \
  --model_size base \
  --per_gpu_batch_size 4 \
  --n_context 10 \
  --name my_experiment \
  --checkpoint_dir checkpoint \
  --eval_freq 500
```  

### Test

[`test.py`](test.py) provides the script to evaluate the performance of the model. An example usage of the script is provided below.

```shell
python test.py \
  --model_path my_model_path \
  --test_data_path my_test_data.json \
  --model_size base \
  --per_gpu_batch_size 4 \
  --n_context 10 \
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
