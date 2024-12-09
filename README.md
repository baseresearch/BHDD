# Burmese Handwritten Digit Dataset (BHDD)

The **Burmese Handwritten Digit Dataset (BHDD)** is a dataset project specifically created for recognizing handwritten Burmese digits. It serves as the Burmese counterpart to the renowned MNIST dataset and is designed to facilitate learning and benchmarking in Machine Learning (ML) and Deep Learning (DL).

![Sample Images](images/sample.png)

---

## Overview

**Dataset Statistics:**
- **Training Set:** 60,000 samples
- **Testing Set:** 27,561 samples
- **Number of Classes:** 10 (Burmese digits 0–9)

**Data Format:**
- **Train Image Shape:** `(60000, 784)`
- **Train Label Shape:** `(60000, 10)`
- **Test Image Shape:** `(27561, 784)`
- **Test Label Shape:** `(27561, 10)`

The dataset was collected from over 150 individuals of different ages (ranging from high school students to professionals in their 50s) and diverse occupations (including clerks, programmers, and others) to achieve a wide variety of handwriting styles. We then preprocessed to mirror the structure and functionality of MNIST.

---

## Dataset Content

The dataset consists of:
1. **Train Images**: 60,000 grayscale images of handwritten Burmese digits, flattened into a 1D array of size 784 (28x28 pixels).
2. **Train Labels**: One-hot encoded labels corresponding to the digit class.
3. **Test Images**: 27,561 grayscale images for testing purposes.
4. **Test Labels**: One-hot encoded labels for testing data.

---

## Installation

Clone this repository to get started:

```bash
$ git clone https://github.com/baseresearch/BHDD.git
$ cd BHDD
```

---

## Usage and Visualizing

We have provided usage and visualization notebooks in this repo. Please check them.

---

## Contribution

We encourage the ML/DL community to contribute by:
- Creating digit recognizers.
- Benchmarking with different models and algorithms.
- Writing tutorials and sharing findings.

For contributions, please fork the repository, make your changes, and submit a pull request.

---

## Citation

If you use the BHDD dataset in your work, please cite this repository:

```
@dataset{bhdd,
  author = {Expa.AI Research Team},
  title = {Burmese Handwritten Digit Dataset (BHDD)},
  year = {2024},
  url = {https://github.com/baseresearch/BHDD}
}
```

---

## Acknowledgments

This dataset would not have been possible without:
- The efforts of the Expa.AI Research Team.
- Volunteers and interns from Taungoo Computer University who contributed handwriting samples.
- Highschool students from St.Augustine / B.E.H.S (2) Kamayut
- Friends and family members of Expa.AI Research Team.
- The community’s ongoing support and interest in ML/DL for the Burmese language.

---

## License

This dataset is released under the [LGPL-3.0 license](LICENSE). Please see the LICENSE file for more details.

---

## Contact

For questions, suggestions, or feedback, please reach out to our team by opening an issue.

---

Let’s build the Burmese ML/DL ecosystem together!

## Note about Git LFS
If you clone this repository without installing Git LFS, you will only fetch the pointer file for the dataset.
In order to get the original dataset, you need to have Git LFS or download the `RAW` file directly.
