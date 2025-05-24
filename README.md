# Mini Projeto 2 — Processamento de Linguagem Natural com Transformers

Este projeto é um estudo aplicado de **modelos de sequência para sequência (Seq2Seq)** usando a biblioteca Hugging Face Transformers, com foco na tarefa de **resumo automático de textos**. Os dados são carregados a partir de um arquivo CSV e avaliados com a métrica **ROUGE**.

## Alunos

- André Luiz Bacelar Gonçalves de Menezes
- Filipe Vasconcelos Moreno
- Gabriel Costa e Silva
- Lucas Silvestre de Barros

## Objetivos

- Aplicar modelos `AutoModelForSeq2SeqLM` da Hugging Face.
- Treinar e avaliar modelos de PLN em GPU.
- Utilizar pipelines de dados com `datasets`, `Trainer`, e `evaluate`.
- Gerar resumos automáticos com `model.generate()`.

## Estrutura do Projeto

```
├── Kaggle/
│   └── kaggle.json     # Credenciais da API do Kaggle
├── data/
│   └── dataset.csv     # Base de dados em CSV
├── notebook.ipynb      # Notebook com o pipeline completo
└── README.md           # Este arquivo
```

## Bibliotecas Usadas

- [Transformers](https://huggingface.co/transformers/)
- [Datasets](https://huggingface.co/docs/datasets/)
- [Evaluate](https://huggingface.co/docs/evaluate/)
- Torch, Pandas, NumPy, SentencePiece
- Kaggle & KaggleHub

## Instalação

```bash
pip install -U transformers datasets rouge-score nltk sentencepiece kaggle kagglehub evaluate torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Configurar Kaggle

Coloque o seu `kaggle.json` com as credenciais da API na pasta `.kaggle/`.

```bash
mkdir ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Dataset

O projeto utiliza um dataset em formato **CSV**, carregado com:

```python
from datasets import load_dataset
dataset = load_dataset("csv", data_files={"train": "data/dataset.csv"})
```

## Modelo e Treinamento

O modelo utilizado segue o padrão `AutoModelForSeq2SeqLM` da Hugging Face. Embora o nome exato não tenha sido detectado, geralmente são usados modelos como `t5-small`, `facebook/bart-base`, entre outros.

```python
from transformers import AutoTokenizer, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("t5-small")  # exemplo
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
```

## Avaliação

O projeto utiliza **ROUGE** como métrica principal de avaliação:

```python
import evaluate
rouge = evaluate.load("rouge")
```

## Geração de Resumos

A geração de texto é feita com:

```python
outputs = model.generate(inputs, max_length=60)
```

## Requisitos

- Python 3.8+
- GPU com CUDA (recomendado)
- Credenciais do Kaggle (opcional, dependendo da origem dos dados)

## Licença

Distribuído sob a Licença MIT. Consulte `LICENSE` para mais detalhes.
