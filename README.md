# Addition with Transformers

This repository contains a solution to the following problem:

How to train an LLM so that it is able to perform addition of large numbers?

## Review of Existing Approaches

### 1\. 

Using a simple prompt (for example, "34526178 + 351678902") is suboptimal and would yield unexpected results, since every unique number string would be represented as a token (or an arbitrary number of tokens, depending on the model's tokenizer). The training dataset would require millions of entries to account for every possible number string. 

The most obvious solution would be to refer to [(Muffo et al., 2023)](#Muffo). The proposed approach suggests using number decomposition for performing arithmetic operations:  

```
2503 = 2 thousands, 5 hundreds, 0 tens, 3 units
```

According to [(Muffo et al., 2023)](#Muffo), Calculon (GPT-2 trained on a dataset created with the number decomposition method) shows accuracy of 0.801 and 0.729 on four- and five-digit numbers respectively.

### 2\. 

Number decomposition is a promising perspective, although the scores achieved by [(Muffo et al., 2023)](#Muffo) are not enough for the task that entails addition of numbers with a large number of digits. [(Nogueira et al., 2021)](#Nogueira) report that the most effective method of number representation for arithmetic operations is 10e-based:

```
832 = 8 10e2 3 10e1 2 10e0
```

Instead of denoting each position with an arbitrary number of words (e. g. "thousands" vs. "tens of thousands"), the 10e-based notation uses only one token per position.

According to [(Nogueira et al., 2021)](#Nogueira), T5 models, after being trained on a sufficient amount of data with 10e-based notation, succeed in adding numbers with up to 60 digits in length with >0.99 accuracy. Moreover, [(Nogueira et al., 2021)](#Nogueira) report that T5 models are capable of extrapolation, i. e. adequately performing addition of numbers with up to 60 digits after being trained on numbers with up to 50 digits. T5-3B, T5-770M, and T5-220M show accuracy of 0.988, 0.442, and 0.862 respectively.

### 3\.  

[(Qian et al., 2021)](#Qian) confirm that prompts without positional encoding are suboptimal, since the self-attention mechanism in Transformers poorly handles repeating digits. For example, it fails to differentiate "1"s in the sequence "67111111111119".

[(Qian et al., 2021)](#Qian) provide a method (LM with Tutor) that is capable of extrapolation: after training the model on 15 examples of up to 5 digits, it succeeds on the task of adding numbers with up to 30 digits, achieving 100% accuracy. The paper mentions that the method uses prompts containing fine-grained action sequences accurately describing the algorithm of the addition operation, with pinpointing where the current digit (the one the model should be looking at) originates from.

However, the description of the proposed method is vague, and [(Qian et al., 2021)](#Qian) provide no source code for recreating the experiments.

### 4\. 

The method closest to the one described in [(Qian et al., 2021)](#Qian) is proposed in [(Recchia, 2021)](#Recchia), albeit not for addition, but for determining division remainders. According to [(Recchia, 2021)](#Recchia), after fine-tuning GPT-Neo with 125M parameters on 200 elaborate demonstrations of solving division problems and reporting the remainders, the model achieves over 80% accuracy.

## Technical Report 1

Training pipeline: [Addition_with_T5.ipynb](https://github.com/entrapolarity/Addition-with-Transformers/blob/main/Addition_with_T5.ipynb).

Due to the RAM and time limit of Google Colab, I decided to work with T5-220M. The solution is based on the [code](https://github.com/castorini/transformers-arithmetic) by [(Nogueira et al., 2021)](#Nogueira). On the task of adding numbers with up to 30 digits, accuracy of **0.94** has been achieved on the test dataset.

### Datasets

The datasets were randomly generated and have a roughly equal proportion of d-digit numbers, where d ∈ \[2, 30].

| Dataset     | Size         |
|:------------|:-------------|
| Training    | 10000 items  |
| Validation  | 200 items    |

Each dataset contains prompts and answers in the following format:

Prompt: 

```
What is 5 10e3 8 10e2 2 10e1 9 10e0 plus 4 10e5 8 10e4 8 10e3 8 10e2 2 10e1 2 10e0?
```

Answer: 

```
4 10e5 9 10e4 4 10e3 6 10e2 5 10e1 1 10e0
```

Hyperparameters:

| Hyperparameter  | Value  |
|:----------------|:-------|
| Optimizer       | AdamW  |
| Learning rate   | 3e-4   |
| Weight decay    | 5e-5   |
| Batch size      | 16     |
| Epochs          | 13     |

## Technical Report 2

Training pipeline: [Addition_with_Demonstration_and_GPT_Neo.ipynb](https://github.com/entrapolarity/Addition-with-Transformers/blob/main/Addition_with_Demonstration_and_GPT_Neo.ipynb).

Acccording to [(Nogueira et al., 2021)](#Nogueira), extrapolation on the addition task is hardly achieved when trained on numbers with fewer than 50 digits, regardless of the model size. In order to overcome the extrapolation problem, I have written a [script](https://github.com/entrapolarity/Addition-with-Transformers/blob/main/Data_Generation.ipynb) for creating datasets with elaborate demonstrations for solving addition problems, inspired by the script for division problems by [(Recchia, 2021)](#Recchia).

The explanation of the demonstrations is available in the file [Data_Generation_Rules.pdf](https://github.com/entrapolarity/Addition-with-Transformers/blob/main/Data_Generation_Rules.pdf), and the visualization of a demonstration example is available in the file [Visualization.pdf](https://github.com/entrapolarity/Addition-with-Transformers/blob/main/Visualization.pdf).

### Datasets

The datasets were randomly generated and have a roughly equal proportion of d-digit numbers, where d ∈ \[2, 30].

| Dataset     | Size         |
|:------------|:-------------|
| Training    | 200 items    |
| Validation  | 50 items     |

The datasets are available in this repository as .txt files.

Hyperparameters:

| Hyperparameter  | Value  |
|:----------------|:-------|
| Optimizer       | AdamW  |
| Learning rate   | 5e-5   |
| Batch size      | 1      |
| Epochs          | 50     |

Despite being trained on a dataset with as few as 200 addition problems, the model achieves an accuracy value of **0.88**. According to [(Nogueira et al., 2021)](#Nogueira) (Appendix D), the best accuracy achieved by training T5-220M on a dataset with 1000 items for 200 epochs for this task is only **<0.7**. 

However, the result documented in the paper is not directly comparable to the result of this experiment, since the test datasets used in the paper are not balanced (i. e. do not have a roughly equal proportion of d-digit numbers), but random (approximately 90% of the numbers havе D-digits, 9% have (D−1)-digits, etc.), and therefore do not conform to the specifications of this task.

In order to compare the results of this experiment to the ones achieved by [(Nogueira et al., 2021)](#Nogueira), I trained the resulting model on 200 more addition problems with numbers having up to 40 digits, in order to artificially increase the number of addition problems with 20-30 digits. Then, I evaluated it on an **unbalanced** (randomly) generated test dataset with numbers having up to 30 digits, achieving accuracy of **0.80**.

Due to the limitations of Google Colab, I propose judging this result not by its absolute accuracy value, but relatively to the results achieved by [(Nogueira et al., 2021)](#Nogueira). Albeit the extrapolation issue has not been overcome, a higher accuracy value has been achieved despite the model having almost half as many parameters, having undergone training on half as many epochs, and the dataset being significantly smaller than in [(Nogueira et al., 2021)](#Nogueira). Final comparison:

|              | Nogueira et al., 2021 | This experiment  |
|:-------------|:----------------------|:-----------------|
| Model        | T5                    | GPT-Neo          |
| Parameters   | 220M                  | 125M             |
| Dataset size | 1000                  | 400 (200 + 200)  |
| Epochs       | 200                   | 100 (50 + 50)    |
| Accuracy     | <0.70                 | **0.80**         |

## References

<a name="Muffo"></a>

**Muffo et al., 2023** - Matteo Muffo, Aldo Cocco, and Enrico Bertino. 2023. [Evaluating transformer language models on arithmetic operations using number decomposition.](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.30.pdf) *arXiv preprint arXiv:2304.10977.*

<a name="Nogueira"></a>
  
**Nogueira et al., 2021** - Rodrigo Nogueira, Zhiying Jiang, and Jimmy Lin. 2021. [Investigating the limitations of transformers with simple arithmetic tasks.](https://arxiv.org/pdf/2102.13019.pdf) *arXiv preprint arXiv:2102.13019.*

<a name="Qian"></a>

**Qian et al., 2021** - Jing Qian, Hong Wang, Zekun Li, Shiyang Li, and Xifeng Yan. 2022. [Limitations of language models in arithmetic and symbolic induction.](https://arxiv.org/pdf/2208.05051.pdf) *arXiv preprint arXiv:2208.05051.*

<a name="Recchia"></a>

**Recchia, 2021** - Gabriel Recchia. 2021. [Teaching autoregressive language models complex tasks by demonstration.](https://arxiv.org/ftp/arxiv/papers/2109/2109.02102.pdf) *arXiv preprint arXiv:2109.02102.* 
