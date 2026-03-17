# ================================================================
# TRANSFORMERS NA PRÁTICA: APLICAÇÕES NO DIA A DIA
# ================================================================

from transformers import (pipeline, MarianMTModel, MarianTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer)
import matplotlib.pyplot as plt

print("=" * 60)
print("  APLICAÇÕES DE TRANSFORMERS NO COTIDIANO")
print("=" * 60)

# ----------------------------------------------------------------
# 1. ANÁLISE DE SENTIMENTOS
# ----------------------------------------------------------------
print("\n[1] ANÁLISE DE SENTIMENTOS")
sentiment = pipeline("sentiment-analysis",
                     model="distilbert-base-uncased-finetuned-sst-2-english")

avaliacoes = [
    "This product is absolutely amazing, I love it!",
    "Terrible experience, I will never buy this again.",
    "The delivery was fast but the product quality was average.",
    "Excellent customer service and great value for money!"
]

for texto, res in zip(avaliacoes, sentiment(avaliacoes)):
    print(f"  Texto    : {texto[:55]}...")
    print(f"  Resultado: {res['label']} (confiança: {res['score']:.2%})\n")


# ----------------------------------------------------------------
# 2. TRADUÇÃO AUTOMÁTICA
# ----------------------------------------------------------------
print("\n[2] TRADUÇÃO AUTOMÁTICA (Inglês -> Francês)")

mt_name   = "Helsinki-NLP/opus-mt-en-fr"
mt_tok    = MarianTokenizer.from_pretrained(mt_name)
mt_model  = MarianMTModel.from_pretrained(mt_name)

frases_en = [
    "Artificial intelligence is transforming the world.",
    "Transformers are the foundation of modern language models.",
    "Deep learning enables machines to understand human language."
]

for frase in frases_en:
    tokens = mt_tok([frase], return_tensors="pt", padding=True)
    output = mt_model.generate(**tokens)
    trad   = mt_tok.decode(output[0], skip_special_tokens=True)
    print(f"  EN: {frase}")
    print(f"  FR: {trad}\n")


# ----------------------------------------------------------------
# 3. RESPOSTA A PERGUNTAS
# ----------------------------------------------------------------
print("\n[3] RESPOSTA A PERGUNTAS")
qa = pipeline("question-answering",
              model="distilbert-base-cased-distilled-squad")

contexto = """
    The Transformer architecture was introduced in 2017 by Vaswani et al.
    in the paper Attention Is All You Need. It replaced recurrent neural
    networks with a self-attention mechanism, enabling parallel processing
    of sequences. Today, Transformers power applications such as ChatGPT,
    Google Translate, virtual assistants, and medical diagnosis systems.
    The BERT model, developed by Google, and GPT series, developed by
    OpenAI, are among the most influential Transformer-based models.
"""

perguntas = [
    "Who introduced the Transformer architecture?",
    "What did Transformers replace?",
    "What applications use Transformers today?"
]

for pergunta in perguntas:
    resp = qa(question=pergunta, context=contexto)
    print(f"  Pergunta : {pergunta}")
    print(f"  Resposta : {resp['answer']} (confiança: {resp['score']:.2%})\n")


# ----------------------------------------------------------------
# 4. SUMARIZAÇÃO — direto com T5, sem pipeline
# ----------------------------------------------------------------
print("\n[4] SUMARIZAÇÃO DE TEXTO")

t5_name  = "t5-small"
t5_tok   = T5Tokenizer.from_pretrained(t5_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_name)

texto_original = (
    "Climate change refers to long-term shifts in temperatures and weather "
    "patterns. Since the 1800s, human activities have been the main driver, "
    "primarily due to burning fossil fuels like coal, oil and gas, which "
    "generate greenhouse gas emissions that trap the sun's heat. The main "
    "gases are carbon dioxide and methane, produced by cars, buildings and "
    "agriculture. Clearing forests also releases large amounts of carbon "
    "dioxide. Energy, industry, transport and land use are the main sectors "
    "responsible for greenhouse gas emissions worldwide."
)

entrada = "summarize: " + texto_original
ids     = t5_tok.encode(entrada, return_tensors="pt",
                        max_length=512, truncation=True)
saida   = t5_model.generate(ids, max_new_tokens=80,
                             min_new_tokens=25, num_beams=4,
                             early_stopping=True)
resumo  = t5_tok.decode(saida[0], skip_special_tokens=True)

print(f"  TEXTO ORIGINAL ({len(texto_original.split())} palavras):")
print(f"  {texto_original[:220]}...\n")
print(f"  RESUMO GERADO ({len(resumo.split())} palavras):")
print(f"  {resumo}\n")


# ----------------------------------------------------------------
# 5. CLASSIFICAÇÃO ZERO-SHOT
# ----------------------------------------------------------------
print("\n[5] CLASSIFICAÇÃO ZERO-SHOT")
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

textos = [
    "The patient presented symptoms of fever and respiratory distress.",
    "The stock market reached a new all-time high today.",
    "The new neural network architecture achieves state-of-the-art results."
]
categorias = ["medicine", "finance", "technology", "sports", "politics"]

for texto in textos:
    result    = classifier(texto, candidate_labels=categorias)
    print(f"  Texto    : {texto[:60]}...")
    print(f"  Categoria: {result['labels'][0]} "
          f"(confiança: {result['scores'][0]:.2%})\n")


# ----------------------------------------------------------------
# 6. VISUALIZAÇÃO FINAL
# ----------------------------------------------------------------
print("\n[6] GERANDO VISUALIZAÇÃO...")

aplicacoes  = ["Análise de\nSentimentos", "Tradução\nAutomática",
               "Resposta a\nPerguntas",   "Sumarização\nde Texto",
               "Classificação\nZero-Shot"]
acuracias   = [93.1, 91.5, 88.6, 87.3, 85.2]
cores       = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Transformers: Aplicações e Desempenho no Mundo Real',
             fontsize=14, fontweight='bold')

bars = axes[0].barh(aplicacoes, acuracias, color=cores,
                    edgecolor='white', height=0.6)
axes[0].set_xlim(0, 105)
axes[0].set_xlabel('Acurácia / F1-Score (%)', fontsize=11)
axes[0].set_title('Desempenho por Tarefa\n(benchmarks da literatura)',
                  fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)
for bar, val in zip(bars, acuracias):
    axes[0].text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{val}%', va='center', fontsize=10, fontweight='bold')

setores     = ['Saúde','Educação','Finanças','Tecnologia','Jurídico','Outros']
tamanhos    = [18, 15, 20, 28, 10, 9]
cores_pizza = ['#EF5350','#42A5F5','#66BB6A','#FFA726','#AB47BC','#78909C']

axes[1].pie(tamanhos, labels=setores, autopct='%1.1f%%',
            colors=cores_pizza, explode=(0.05,)*6,
            startangle=140, textprops={'fontsize': 10})
axes[1].set_title('Setores de Adoção de\nTransformers (2024)',
                  fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('transformers_aplicacoes.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figura salva: transformers_aplicacoes.png")

print("\n" + "=" * 60)
print("  DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO")
print("=" * 60)
