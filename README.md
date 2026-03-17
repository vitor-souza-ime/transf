# Transformers na Prática: Aplicações no Dia a Dia

Este repositório apresenta um **exemplo prático de aplicações da arquitetura Transformer**, utilizando a biblioteca [Hugging Face Transformers](https://huggingface.co/transformers/). O objetivo é demonstrar como modelos pré-treinados podem ser aplicados em tarefas do cotidiano de forma simples e eficiente.

---

## Funcionalidades

O script `transformers_demo.py` realiza as seguintes aplicações:

1. **Análise de Sentimentos**  
   Classifica avaliações de produtos ou textos em sentimentos positivos ou negativos usando `distilbert-base-uncased-finetuned-sst-2-english`.

2. **Tradução Automática**  
   Traduz textos do inglês para o francês utilizando o modelo `Helsinki-NLP/opus-mt-en-fr`.

3. **Resposta a Perguntas**  
   Responde perguntas baseadas em um contexto fornecido, usando `distilbert-base-cased-distilled-squad`.

4. **Sumarização de Texto**  
   Gera resumos de textos longos diretamente com o modelo T5 (`t5-small`), sem pipeline.

5. **Classificação Zero-Shot**  
   Classifica textos em categorias pré-definidas sem necessidade de treinamento adicional, usando `facebook/bart-large-mnli`.

6. **Visualização de Resultados**  
   Cria gráficos comparativos mostrando desempenho das aplicações e setores que adotam Transformers em 2024.

---

## Instalação

1. Clone este repositório:

```bash
git clone https://github.com/seu-usuario/transformers-na-pratica.git
cd transformers-na-pratica
````

2. Crie um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

> **requirements.txt** sugerido:
>
> ```
> transformers
> torch
> matplotlib
> ```

---

## Uso

Execute o script principal:

```bash
python transformers_demo.py
```

O script exibirá no terminal os resultados de cada aplicação e gerará uma **figura `transformers_aplicacoes.png`** com visualizações de desempenho e setores de adoção.

---

## Estrutura do Repositório

```
transformers-na-pratica/
│
├── transformers_demo.py   # Script principal com todas as demonstrações
├── transformers_aplicacoes.png  # Figura gerada pelo script
├── requirements.txt       # Dependências do projeto
└── README.md              # Documentação do projeto
```

---

## Contribuição

Contribuições são bem-vindas!
Sinta-se à vontade para abrir *issues*, sugerir melhorias ou enviar *pull requests*.

---

## Licença

Este projeto está licenciado sob a **MIT License**.

---

💡 **Observação:** Este projeto é educativo e serve para demonstrar aplicações práticas de Transformers no cotidiano.
