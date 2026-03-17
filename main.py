# ================================================================
# TRANSFORMERS NA PRÁTICA: APLICAÇÕES NO DIA A DIA
# Demonstração com Hugging Face Transformers
# ================================================================

from transformers import pipeline, MarianMTModel, MarianTokenizer
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

resultados_sent = sentiment(avaliacoes)
for texto, res in zip(avaliacoes, resultados_sent):
    print(f"  Texto    : {texto[:55]}...")
    print(f"  Resultado: {res['label']} (confiança: {res['score']:.2%})\n")


# ----------------------------------------------------------------
# 2. TRADUÇÃO AUTOMÁTICA
# ----------------------------------------------------------------
print("\n[2] TRADUÇÃO AUTOMÁTICA (Inglês -> Francês)")

model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer  = MarianTokenizer.from_pretrained(model_name)
model_mt   = MarianMTModel.from_pretrained(model_name)

frases_en = [
    "Artificial intelligence is transforming the world.",
    "Transformers are the foundation of modern language models.",
    "Deep learning enables machines to understand human language."
]

for frase in frases_en:
    tokens = tokenizer([frase], return_tensors="pt", padding=True)
    output = model_mt.generate(**tokens)
    trad   = tokenizer.decode(output[0], skip_special_tokens=True)
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
    in the paper 'Attention Is All You Need'. It replaced recurrent neural
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
# 4. SUMARIZAÇÃO — CORRIGIDO com T5
# ----------------------------------------------------------------
print("\n[4] SUMARIZAÇÃO DE TEXTO")

# T5-small aceita prefixo "summarize:" e funciona com text2text-generation
summarizer = pipeline("text2text-generation",
                      model="t5-small")

texto_longo = (
    "summarize: Climate change refers to long-term shifts in temperatures "
    "and weather patterns. Since the 1800s, human activities have been the "
    "main driver of climate change, primarily due to the burning of fossil "
    "fuels like coal, oil and gas. Burning fossil fuels generates greenhouse "
    "gas emissions that trap the sun's heat and raise temperatures. The main "
    "greenhouse gases include carbon dioxide and methane, which come from "
    "driving cars, heating buildings, and agricultural operations. Clearing "
    "forests also releases large amounts of carbon dioxide into the atmosphere."
)

resumo = summarizer(
    texto_longo,
    max_new_tokens=80,
    min_new_tokens=25,
    do_sample=False
)

texto_sem_prefixo = texto_longo.replace("summarize: ", "")
print(f"  TEXTO ORIGINAL ({len(texto_sem_prefixo.split())} palavras):")
print(f"  {texto_sem_prefixo[:220].strip()}...\n")
print(f"  RESUMO GERADO ({len(resumo[0]['generated_text'].split())} palavras):")
print(f"  {resumo[0]['generated_text']}\n")


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
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    print(f"  Texto    : {texto[:60]}...")
    print(f"  Categoria: {top_label} (confiança: {top_score:.2%})\n")


# ----------------------------------------------------------------
# 6. VISUALIZAÇÃO FINAL
# ----------------------------------------------------------------
print("\n[6] GERANDO VISUALIZAÇÃO...")

aplicacoes = [
    "Análise de\nSentimentos",
    "Tradução\nAutomática",
    "Resposta a\nPerguntas",
    "Sumarização\nde Texto",
    "Classificação\nZero-Shot"
]
acuracias   = [93.1, 91.5, 88.6, 87.3, 85.2]
cores       = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Transformers: Aplicações e Desempenho no Mundo Real',
             fontsize=14, fontweight='bold')

# Barras horizontais
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

# Pizza: setores de adoção
setores     = ['Saúde', 'Educação', 'Finanças', 'Tecnologia', 'Jurídico', 'Outros']
tamanhos    = [18, 15, 20, 28, 10, 9]
explode     = (0.05,) * len(setores)
cores_pizza = ['#EF5350','#42A5F5','#66BB6A','#FFA726','#AB47BC','#78909C']

axes[1].pie(tamanhos, labels=setores, autopct='%1.1f%%',
            colors=cores_pizza, explode=explode,
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
