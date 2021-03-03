# Prototypical Networks
A re-implementation of "Prototypical Networks for Few-shot Learning" by Jake Snell, Kevin Swersky, and Richard S. Zemel.

See: https://arxiv.org/abs/1703.05175

## Results
| n-way  | k-shot | Accuracy | Original Paper |
| ------ | ------ | -------- | -------------- |
| 5      | 1      | 95.1%    | 98.8% |
| 5      | 5      | 98.6%    | 99.7% |
| 20     | 1      | 94.7%    | 96.0% |
| 20     | 5      | 95.8%    | 98.9% |

## Setup
Using Python 3.9.1, run `pip install -r requirements.txt` and then `python prototypical.py` to train and evaluate the model.

See `python prototypical.py --help` for command line arguments.
