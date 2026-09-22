# The Harder Text Embedding Benchmark (HTEB) v1.1

HTEB evaluates text embedding models on original and LLM-transformed inputs to measure robustness across three axes using eight deployment-oriented transformations:

| Axis              | Transformation         | Description                                                                                                                                                            |
| ----------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lexical/Stylistic | `paraphrasing`         | Rewrites the text while preserving its meaning.                                                                                                                        |
| Lexical/Stylistic | `backtranslation`      | Translates the text into a randomly selected intermediate language, excluding the source language, and then back to the original language while aiming to preserve its meaning. |
| Lexical/Stylistic | `style_change`         | Changes formal language to informal language, or vice versa, while preserving the meaning of a text.                                                                   |
| Length            | `expansion`            | Expands the text into a longer version while preserving its core meaning. Intended to significantly increase text length.                                               |
| Length            | `summarise`            | Reduces the text into a shorter version while preserving its core meaning. Intended to significantly decrease text length.                                              |
| Length            | `summarised_expansion` | First expands the text, then summarises the expanded version intending to increase text length but less so than `expansion`.                                            |
| Language          | `translation`          | Translates texts into a randomly selected target language. Uses the same language for paired texts.        |
| Language          | `cross_translation`    | Translates each text into an independently selected random target language. Allows paired texts to be in different languages.                                          |

The exact transformation prompts can be found in [hteb/prompts.py](hteb/prompts.py). This v1.1 implementation is a further development of HTEB's original implementation. Particularly it now uses [cyankiwi/gemma-4-31B-it-AWQ-4bit](https://huggingface.co/cyankiwi/gemma-4-31B-it-AWQ-4bit) as the transformation model by default.

---
## Installation

HTEB requires **Linux**, **Python 3.10** and **uv**. From the repository root, install the locked dependencies:

```bash
uv sync --locked
```

The environment includes evaluation and generation libraries, including vLLM and PyTorch with CUDA 13.0. Choose models, batch sizes and generation limits that fit your hardware.

If a model or dataset requires Hugging Face authentication, set `HF_TOKEN` in your shell or add `HF_TOKEN=hf_your_token_here` to a local `.env` file. HTEB loads `.env` from the working directory and preserves existing shell values. Keep credentials out of Git; `.env` is ignored.

---
## Running HTEB

Run all commands from the repository root. HTEB is configured through configuration files in [configs](configs/) for which we provide four examples:

| Configuration                                                | # datasets | # embedding models | # seeds | Description                                                                                                                                                                                        |
| ------------------------------------------------------------ | ---------- | ------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Banking77](configs/example_banking77.yaml)                  | 1          | 4                  | 3       | Example dataset for which transformations are included [here](data/transformations/cyankiwi~2Fgemma-4-31B-it-AWQ-4bit/banking77). The evaluation can be run without expensive generations of transformations. |
| [Seven-dataset example](configs/example_seven_datasets.yaml) | 7          | 4                  | 3       | An example config with one dataset per task to broadly test HTEB.                                                                                                                                             |
| [HTEB English](configs/hteb_english.yaml)                    | 19         | 4                  | 3       | Config to run HTEB on its English datasets (matches the datasets used in the paper but model scope is still reduced to reduce the required compute and make running it more practical).                                                                                                                                            |
| [HTEB Multilingual](configs/hteb_multilingual.yaml)          | 13         | 4                  | 3       | Config to run HTEB on its Multilingual datasets (matches the datasets used in the paper but model scope is still reduced to reduce the required compute and make running it more practical).                                                                                                                                       |

Start with Banking77. Its default reuse mode (`load_existing_transformations: true`) reuses the prepared files under `data/transformations/cyankiwi~2Fgemma-4-31B-it-AWQ-4bit/banking77/`.

```bash
uv run --locked hteb configs/example_banking77.yaml
```

With complete prepared files and `load_existing_transformations: true`, the generator is not loaded. Keep `generator_model` unchanged to select the matching transformation folder. Embedding models are still loaded and downloaded when needed.

For broader evaluations run one of the following (these will require the generation of the respective transformations):

```bash
uv run --locked hteb configs/example_seven_datasets.yaml
uv run --locked hteb configs/hteb_english.yaml
uv run --locked hteb configs/hteb_multilingual.yaml
```

Each configuration evaluates four lightweight embedding models: multilingual MPNet, Jina nano, Harrier 0.6B and Qwen3 Embedding 0.6B. These are evaluated across all eight transformations and three random seeds.

---
## Configuration

To specify HTEB's experiments, adjust the existing YAML configuration or set up a new one. The config particularly specifies the following parameters among others:

- **Models:** Set the transformations generation model using `generator_model` and  the embedding models to be evaluated by `emb_models[].model_id`. Both accept Hugging Face Hub IDs. Generation uses vLLM; embedding evaluation uses SentenceTransformers. Model-specific options, including batch size, `prompt_names` and task-specific `encode_kwargs`, belong under each embedding model's `settings`.
- **Datasets and transformations:** Edit `selection.datasets` and `selection.transformations`. Dataset selection accepts individual names, `english`, `multilingual`, or `[english, multilingual]`. [Dataset metadata](hteb/dataset_metadata/) defines all supported datasets.
- **Seeds:** Choose nonnegative integer values for `selection.seed_transform` and `evaluation.seed_evaluation`. The three seeds in the example configs are examples only. Entries pair by position, so both lists must have equal length. Reusing transformations requires saved files for the selected transformation seeds.
- **Device:** The example configs set `devices: [0]` for `generation` and `evaluation`. To use two GPUs, for example, set `devices: [0, 1]`.
- **Generation:** adjust `generation.batch_size`, `max_model_length_default`, `max_tokens_default` and `max_tokens_expansion_default` as needed. Set `generation.transform_corpus: false` to leave retrieval documents and reranking candidates unchanged. Generation prompts are defined in [hteb/prompts.py](hteb/prompts.py).
- **Loading existing transformations:**
	- `load_existing_transformations: true` loads transformations from disk. The transformations must already have been generated and saved to disk. No generator is needed.
	- `load_existing_transformations: false` downloads the selected originals as needed, generates the entire selection and evaluates the new files.

Transformations are saved as paired `data_<hash>.parquet` and `metadata_<hash>.JSON` files under `data/transformations/<generator-model>/<dataset>/<transformation>/<seed>/`.  Parquet files contain originals and transformed records. Reuse selects the newest pair by recorded generation time and rejects incomplete or ambiguous data. Changing generation settings in reuse mode does not alter saved transformations.

---
## Results

Successful runs print score tables and save:

```text
results/<automatic-run-name>/
├── scores.JSON
├── settings.json
└── run_log.txt
```

`scores.JSON` contains task scores and data-quality details. `settings.json` records validated settings, detected model versions and evaluation inputs. `run_log.txt` records progress and diagnostics.

Scores are task metrics multiplied by 100. Tables average scores over seed pairs, then selected transformations within each axis. **HTEB Total** averages the selected axes with equal weight; **Original** is reported separately and is excluded from that total. The overall table also averages across datasets. Compare runs using the same datasets, transformations and evaluation settings.

Invalid or truncated generated responses are retried. Fields still failing after retries are saved as empty text, with fallback counts in transformation metadata. Evaluation fills missing transformed text from the originals in memory without rewriting the saved data. Originals and transformations use the same retained rows, with exclusions reported. Review these diagnostics alongside scores; fatal backend or write errors stop the run.

---
## Citation

The project is based on our HTEB paper. When using or building on HTEB, please cite it.

**EMNLP 2026 (to appear):**

```bibtex
@inproceedings{FrankAfli2026_HTEB_EMNLP,
  title = {The {Harder Text Embedding Benchmark} ({HTEB}): Beyond One-dimensional Static Robustness},
  author = {Frank, Manuel and Afli, Haithem},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year = {2026},
  publisher = {Association for Computational Linguistics},
  note = {To appear},
}
```

**Preprint:**

```bibtex
@misc{FrankAfli2026_HTEB_Preprint,
  title = {The {Harder Text Embedding Benchmark} ({HTEB}): Beyond One-dimensional Static Robustness},
  author = {Frank, Manuel and Afli, Haithem},
  year = {2026},
  month = may,
  journal = {arXiv.org},
  url = {https://arxiv.org/abs/2605.28190},
}
```

---
## License

HTEB's code is available under the [Apache License 2.0](LICENSE). Dataset and model licences are set by their respective providers.

### Transformation model: cyankiwi/gemma-4-31B-it-AWQ-4bit

The default transformation model, [cyankiwi/gemma-4-31B-it-AWQ-4bit](https://huggingface.co/cyankiwi/gemma-4-31B-it-AWQ-4bit), is licensed under the [Apache License 2.0](https://ai.google.dev/gemma/apache_2). For ethical considerations please refer to the [model card](https://huggingface.co/cyankiwi/gemma-4-31B-it-AWQ-4bit).

### Dataset example: BANKING77

HTEB provides transformations as outlined above of the BANKING77 dataset as an example for using HTEB. BANKING77 itself is released by [PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets#banking).

**Please cite their paper when using the original or transformed BANKING77 dataset:**
```bibtex
@inproceedings{casanueva-etal-2020-efficient,
    title = "Efficient Intent Detection with Dual Sentence Encoders",
    author = "Casanueva, I{\~n}igo  and
      Tem{\v{c}}inas, Tadas  and
      Gerz, Daniela  and
      Henderson, Matthew  and
      Vuli{\'c}, Ivan",
    editor = "Wen, Tsung-Hsien  and
      Celikyilmaz, Asli  and
      Yu, Zhou  and
      Papangelis, Alexandros  and
      Eric, Mihail  and
      Kumar, Anuj  and
      Casanueva, I{\~n}igo  and
      Shah, Rushin",
    booktitle = "Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.nlp4convai-1.5/",
    doi = "10.18653/v1/2020.nlp4convai-1.5",
    pages = "38--45",
}
```

BANKING77 data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated in the [dataset card](https://huggingface.co/datasets/PolyAI/banking77#licensing-information) and [github repo](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE). HTEB's software licence does not replace these terms. Attribution does not imply endorsement by PolyAI or the original authors. When redistributing HTEB's Banking77 transformations, keep this attribution, the source and licence links, and identify any additional changes.
