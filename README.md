# Interactive Image Retrieval Enhanced by Content and Textual Feedback

## What's All This About?

Think about how your brain effortlessly links attributes to things. When someone says "apple," your mind instantly pictures an apple, whether it's green, red, or something in between.

Now, in the digital world, we usually search for stuff using words or images, but it's time for a twist.

## Sneak Peek

<img align="left" src="static/images/sample.png" width="100%">

We're shaking up the way you search. Imagine blending words and pictures to find what you're looking for. That's what our project is all about. You toss in some text, and it influences the images you get back. Perfect for hunting down products online, keeping an eye on things, or finding stuff on the web.

Here's a quick example to paint the picture. Say you're shopping online and want a dress that's similar to your friend's but with some specific tweaks, like stripes and a bit more coverage. Our smart algorithm makes it happen.

## Model Magic

We've got a nifty model, powered by autoencoders and transformers, to make sense of text and images for your search. It's all about learning from the good stuff and using it to get you the perfect matches. We even throw in a sprinkle of math to keep things in check.
![How It Works](static/images/model_en.png)

## Check Out the Results

Our approach is the champ here, beating the state-of-the-art method TIRG and ComposeAE on the MIT-States benchmark dataset.

<!-- Take a peek at some snazzy retrieval results below: -->
<!-- ![Qual](FIQ_Retrieval.jpg) -->

## What You Need and How to Set It Up

-   You'll find all the packages you need in [requirements.txt](requirements.txt).

## About the Code [(Inspired by ComposeAE)](https://github.com/ecom-research/ComposeAE/blob/master/README.md)

We built our code based on ComposeAE's work, but we've spiced it up with some serious upgrades.

-   `main.py`: The main event – run this script for training and testing.
-   `datasets.py`: Gets the goods, like loading images and whipping up training retrieval queries.
-   `text_model.py`: A brainy LSTM model for text features.
-   `img_text_composition_models.py`: Fancy models for mixing up images and text.
-   `torch_function.py`: Holds our secret sauce – the soft triplet loss function and feature normalization magic.
-   `test_retrieval.py`: This one's all about retrieval tests and calculating recall performance.

## Time to Run Some Experiments

### Grab the Datasets

#### MITStates Dataset

Get your hands on the MITStates dataset right [here](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html). Save it in the `data` folder. Just make sure it's got these files:

`data/processed/mitstates/images/<adj noun>/*.jpg`

#### Fashion200k Dataset

First up, grab the Fashion200k dataset from [this spot](https://github.com/xthan/fashion-200k), and toss it into your trusty `data` folder. To keep things on the level, we're using the same test queries as TIRG. You can nab those queries from [here](https://storage.googleapis.com/image_retrieval_css/test_queries.txt). Just make sure your dataset has these files:

```
data/processed/fashion200k/labels/*.txt
data/processed/fashion200k/women/<category>/<caption>/<id>/*.jpeg
data/processed/fashion200k/test_queries.txt`
```

#### FashionIQ Dataset

Now, for the FashionIQ dataset, head over to [this link](https://github.com/XiaoxiaoGuo/fashion-iq), and stash it in your `data` folder. This one's a bit of a mix, with three separate subsets: `dress`, `top-tee`, and `shirt`. We're taking those two annotations and giving them a little text twist, combining them to make it look more like something a user might ask on an e-commerce platform.

What's more, we're bringing all three categories together for a beefed-up training set and training a single model on it. We're doing the same with the validation sets to keep things neat and tidy.

## Running the Show

To train and test your models, just use the right commands. Here are some examples to get you started:

-   **Training the Original TIRG Model on MITStates Dataset:**

    ```bash
    python -W ignore main.py --dataset=mitstates --dataset_path=../data/mitstates/ --model=tirg --loss=soft_triplet --learning_rate_decay_frequency=50000 --num_iters=160000 --weight_decay=5e-5 --comment=mitstates_tirg_original --log_dir ../logs/mitstates/

    ```

-   **Training TIRG with BERT on MITStates Dataset:**

    ```bash
    python -W ignore main.py --dataset=mitstates --dataset_path=../data/mitstates/ --model=tirg --loss=soft_triplet --learning_rate_decay_frequency=50000 --num_iters=160000 --weight_decay=5e-5 --comment=mitstates_tirg_bert --log_dir ../logs/mitstates/ --use_bert True
    ```

-   **Training TIRG with Complete Text Query on MITStates Dataset:**

    ```bash
    python -W ignore main.py --dataset=mitstates --dataset_path=../data/mitstates/ --model=tirg --loss=soft_triplet --learning_rate_decay_frequency=50000 --num_iters=160000 --weight_decay=5e-5 --comment=mitstates_tirg_complete_text_query --log_dir ../logs/mitstates/ --use_complete_text_query True
    ```

-   **Training the ComposeAE Model on Fashion200k Dataset:**

    ```bash
    python -W ignore main.py --dataset=fashion200k --dataset_path=../data/fashion200k/ --model=composeAE --loss=batch_based_classification --learning_rate_decay_frequency=50000 --num_iters=160000 --use_bert True --use_complete_text_query True --weight_decay=5e-5 --comment=fashion200k_composeAE --log_dir ../logs/fashion200k/
    ```

-   **Training the RealSpaceConcatAE (ComposeAE Model with Concatenation in Real Space) on FashionIQ Dataset:**
    ```bash
    python -W ignore main.py --dataset=fashionIQ --dataset_path=../data/fashionIQ/ --model=RealSpaceConcatAE --loss=batch_based_classification --learning_rate_decay_frequency=8000 --num_iters=100000 --use_bert True --use_complete_text_query True --comment=fashionIQ_RealSpaceConcatAE --log_dir ../logs/fashionIQ/
    ```

This version simplifies the instructions and makes it more approachable for users.

## Important Stuff to Know

### Using the BERT Model

We've got a snazzy BERT model that helps encode text queries. We use BERT-as-service with Uncased BERT-Base, and it dishes out a 512-dimensional feature vector for text queries. To get the nitty-gritty on how to use it, check out the instructions [here](https://github.com/jina-ai/clip-as-service). Just a heads-up, make sure you've got BERT-as-service up and running in the background before you dive into training your models.

### Keep an Eye on Performance with Tensorboard

To keep tabs on how your models are doing, use this command to monitor loss and retrieval performance:
`bash
    tensorboard --logdir ./reports/fashion200k/ --port 8898
    `

### Give Credit Where It's Due

If our code has been a big help in your research, show us some love by citing it:

```
@InProceedings{,
    author    = {},
    title     = {},
    booktitle = {},
    month     = {},
    year      = {},
    pages     = {}
}
```
