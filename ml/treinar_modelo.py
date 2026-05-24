# =============================================================
#  TREINO DA ÁRVORE DE DECISÃO - CARRINHO SEGUIDOR DE LINHA
#  Roda no PC com Python. Gera o modelo e exporta para C++.
#
#  Instalar antes de rodar:
#  pip install scikit-learn pandas matplotlib numpy
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# =============================================================
# PARTE 1 — CRIAR O DATASET
#
# Explicação:
#   Cada linha do dataset representa UMA leitura dos sensores
#   em um momento da corrida. Os sensores TCRT5000 em modo
#   analógico retornam valores de 0 a 1023:
#     ~0–200   = sensor sobre superfície BRANCA (reflexo alto)
#     ~800–1023 = sensor sobre linha PRETA (absorve luz)
#
#   Colunas:
#     s_esq  → leitura analógica do sensor esquerdo (0–1023)
#     s_dir  → leitura analógica do sensor direito  (0–1023)
#     acao   → o que o carrinho DEVE fazer nessa situação
#
#   Ações (rótulos):
#     0 = IR RETO          (linha centrada entre os 2 sensores)
#     1 = CURVA ESQUERDA   (linha desviada para a esquerda)
#     2 = CURVA DIREITA    (linha desviada para a direita)
#     3 = BUSCAR LINHA     (perdeu a linha - ambos no branco)
#     4 = CRUZAMENTO       (ambos os sensores sobre preto)
# =============================================================

def criar_dataset_sintetico():
    """
    Cria dados sintéticos baseados no comportamento físico
    esperado dos sensores TCRT5000 na pista real.

    Cada np.random.randint gera N leituras aleatórias dentro
    da faixa de valores que aquela situação produziria.
    Usamos ruído (variação nos valores) para que a árvore
    aprenda a generalizar, não a memorizar valores exatos.
    """

    print("Criando dataset sintético baseado na física dos sensores...")
    dados = []

    # ----------------------------------------------------------
    # Cenário 0: IR RETO
    #   Ambos os sensores leem valores intermediários-altos
    #   pois a linha passa entre os dois (borda da linha).
    #   Na prática: s_esq e s_dir estão na transição branco/preto.
    # ----------------------------------------------------------
    n = 400  # quantidade de amostras por cenário
    for _ in range(n):
        s_esq = np.random.randint(400, 750)   # borda esquerda da linha
        s_dir = np.random.randint(400, 750)   # borda direita da linha
        dados.append([s_esq, s_dir, 0])       # acao = 0 (reto)

    # ----------------------------------------------------------
    # Cenário 1: CURVA ESQUERDA
    #   O carrinho desviou para a direita, então o sensor
    #   ESQUERDO está sobre a linha (leitura alta) e o
    #   DIREITO está sobre o fundo branco (leitura baixa).
    #   Ação: girar à esquerda para voltar à linha.
    # ----------------------------------------------------------
    for _ in range(n):
        s_esq = np.random.randint(750, 1023)  # sobre a linha preta
        s_dir = np.random.randint(0, 250)     # sobre o fundo branco
        dados.append([s_esq, s_dir, 1])       # acao = 1 (curva esq)

    # ----------------------------------------------------------
    # Cenário 2: CURVA DIREITA
    #   O oposto: sensor DIREITO sobre a linha, ESQUERDO no branco.
    #   Ação: girar à direita.
    # ----------------------------------------------------------
    for _ in range(n):
        s_esq = np.random.randint(0, 250)     # sobre o fundo branco
        s_dir = np.random.randint(750, 1023)  # sobre a linha preta
        dados.append([s_esq, s_dir, 2])       # acao = 2 (curva dir)

    # ----------------------------------------------------------
    # Cenário 3: BUSCAR LINHA (perdeu a linha)
    #   Ambos os sensores estão sobre o fundo branco.
    #   O carrinho saiu completamente da faixa da linha.
    #   Ação: girar devagar para procurar a linha.
    # ----------------------------------------------------------
    for _ in range(n):
        s_esq = np.random.randint(0, 200)     # branco
        s_dir = np.random.randint(0, 200)     # branco
        dados.append([s_esq, s_dir, 3])       # acao = 3 (busca)

    # ----------------------------------------------------------
    # Cenário 4: CRUZAMENTO
    #   Ambos os sensores estão sobre a linha preta ao mesmo tempo.
    #   Isso acontece no cruzamento em X da sua pista.
    #   Ação: seguir reto por um tempo fixo (ignorar sensores).
    #   Nota: usamos leituras bem altas para diferenciar do cenário 0.
    # ----------------------------------------------------------
    for _ in range(n):
        s_esq = np.random.randint(800, 1023)  # preto
        s_dir = np.random.randint(800, 1023)  # preto
        dados.append([s_esq, s_dir, 4])       # acao = 4 (cruzamento)

    # Converte para DataFrame (tabela)
    df = pd.DataFrame(dados, columns=['s_esq', 's_dir', 'acao'])

    # Embaralha as linhas (importante para o treinamento)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset criado: {len(df)} amostras")
    print(df['acao'].value_counts().rename({
        0: 'Reto', 1: 'Curva Esq', 2: 'Curva Dir',
        3: 'Buscar Linha', 4: 'Cruzamento'
    }))
    return df


def carregar_ou_criar_dataset(caminho_csv='dados_pista.csv'):
    """
    Se você já tiver um CSV com dados reais do carrinho/Webots,
    ele carrega esse CSV. Senão, usa o dataset sintético.

    Formato esperado do CSV real:
        s_esq,s_dir,acao
        823,12,1
        45,891,2
        ...
    """
    if os.path.exists(caminho_csv):
        print(f"Carregando dados reais de '{caminho_csv}'...")
        df = pd.read_csv(caminho_csv)
        print(f"Dados reais carregados: {len(df)} amostras")
    else:
        print(f"Arquivo '{caminho_csv}' não encontrado.")
        print("Usando dataset sintético (baseado na física dos sensores).\n")
        df = criar_dataset_sintetico()
        # Salva para você ver/editar
        df.to_csv(caminho_csv, index=False)
        print(f"Dataset salvo em '{caminho_csv}' para você revisar.\n")
    return df


# =============================================================
# PARTE 2 — TREINAR A ÁRVORE DE DECISÃO
#
# Explicação dos parâmetros do DecisionTreeClassifier:
#
#   max_depth=5
#     Limita a profundidade máxima da árvore a 5 níveis.
#     Profundidade maior = mais preciso nos dados de treino,
#     mas pode "memorizar" ruído (overfitting).
#     Profundidade 5 com 2 sensores é mais que suficiente.
#
#   min_samples_leaf=10
#     Cada folha (decisão final) precisa ter pelo menos
#     10 amostras de treino que levem até ela.
#     Evita que a árvore crie regras para casos muito raros.
#
#   criterion='gini'
#     Métrica usada para decidir como dividir os dados em
#     cada nó da árvore. 'gini' é o padrão e funciona bem.
#
#   random_state=42
#     Semente aleatória para resultados reproduzíveis.
#     (Qualquer número serve — 42 é convenção.)
# =============================================================

def treinar_modelo(df):
    print("\n" + "="*50)
    print("TREINANDO A ÁRVORE DE DECISÃO")
    print("="*50)

    # Separar entradas (X) e saída (y)
    X = df[['s_esq', 's_dir']].values   # features: leituras dos sensores
    y = df['acao'].values               # target: ação correta

    # Dividir em treino (80%) e teste (20%)
    # Assim podemos medir se o modelo generaliza bem
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=0.20,      # 20% para testar
        random_state=42,     # reproduzível
        stratify=y           # garante proporção igual de cada classe
    )

    print(f"\nAmostras de treino: {len(X_treino)}")
    print(f"Amostras de teste:  {len(X_teste)}")

    # Criar e treinar o modelo
    modelo = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        criterion='gini',
        random_state=42
    )
    modelo.fit(X_treino, y_treino)

    # Avaliar o modelo
    y_pred = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, y_pred)

    print(f"\n✓ Acurácia no conjunto de TESTE: {acuracia*100:.1f}%")
    print("\nRelatório detalhado por classe:")
    print(classification_report(
        y_teste, y_pred,
        target_names=['Reto', 'CurvaEsq', 'CurvaDir', 'BuscarLinha', 'Cruzamento']
    ))

    # Mostrar as regras da árvore em texto
    print("\n" + "="*50)
    print("REGRAS APRENDIDAS PELA ÁRVORE:")
    print("="*50)
    regras = export_text(modelo, feature_names=['s_esq', 's_dir'])
    print(regras)

    return modelo


# =============================================================
# PARTE 3 — VISUALIZAR A ÁRVORE
#
# Gera uma imagem da árvore de decisão para você entender
# visualmente as regras que ela aprendeu.
# =============================================================

def visualizar_arvore(modelo):
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        modelo,
        feature_names=['Sensor Esq', 'Sensor Dir'],
        class_names=['Reto', 'CurvaEsq', 'CurvaDir', 'Buscar', 'Cruzamento'],
        filled=True,          # cores diferentes por classe
        rounded=True,         # cantos arredondados
        fontsize=10,
        ax=ax
    )
    plt.title('Árvore de Decisão — Seguidor de Linha', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('arvore_decisao.png', dpi=150, bbox_inches='tight')
    print("\n✓ Imagem da árvore salva em 'arvore_decisao.png'")
    plt.show()


# =============================================================
# PARTE 4 — EXPORTAR PARA C++ (ESP32-S3)
#
# Converte as regras da árvore em código C++ puro.
# Esse código não precisa de nenhuma biblioteca de ML no ESP32.
# É simplesmente uma série de if/else gerada automaticamente.
# =============================================================

def exportar_para_cpp(modelo, caminho='modelo_arvore.h'):
    """
    Gera um arquivo .h (header C++) com a função predict()
    que implementa a árvore de decisão treinada.
    Você copia esse arquivo para a pasta do seu projeto Arduino.
    """

    tree = modelo.tree_

    # Extrai os dados internos da árvore treinada
    left     = tree.children_left    # índice filho esquerdo de cada nó
    right    = tree.children_right   # índice filho direito de cada nó
    feature  = tree.feature          # qual feature divide cada nó
    threshold= tree.threshold        # valor de corte de cada nó
    value    = tree.value            # distribuição de classes em cada folha

    nomes_features = ['s_esq', 's_dir']
    nomes_acoes    = ['RETO', 'CURVA_ESQ', 'CURVA_DIR', 'BUSCAR_LINHA', 'CRUZAMENTO']

    linhas = []
    linhas.append("// ============================================================")
    linhas.append("// MODELO DE ÁRVORE DE DECISÃO — GERADO AUTOMATICAMENTE")
    linhas.append("// Treinado com scikit-learn DecisionTreeClassifier")
    linhas.append("// Cole este arquivo na pasta do seu projeto Arduino/ESP32")
    linhas.append("// ============================================================")
    linhas.append("")
    linhas.append("#ifndef MODELO_ARVORE_H")
    linhas.append("#define MODELO_ARVORE_H")
    linhas.append("")
    linhas.append("// Ações possíveis")
    linhas.append("#define ACAO_RETO         0")
    linhas.append("#define ACAO_CURVA_ESQ    1")
    linhas.append("#define ACAO_CURVA_DIR    2")
    linhas.append("#define ACAO_BUSCAR_LINHA 3")
    linhas.append("#define ACAO_CRUZAMENTO   4")
    linhas.append("")
    linhas.append("// Função gerada pela árvore de decisão treinada")
    linhas.append("// s_esq: leitura analógica do sensor esquerdo (0–1023)")
    linhas.append("// s_dir: leitura analógica do sensor direito  (0–1023)")
    linhas.append("// retorna: código da ação a executar")
    linhas.append("int predict(int s_esq, int s_dir) {")

    def gerar_no(no_idx, profundidade):
        """Percorre a árvore recursivamente e gera o código C++"""
        indent = "  " * (profundidade + 1)

        # Nó folha: retorna a classe majoritária
        if left[no_idx] == -1:
            classe = int(np.argmax(value[no_idx][0]))
            nome   = nomes_acoes[classe] if classe < len(nomes_acoes) else str(classe)
            linhas.append(f"{indent}return {classe}; // {nome}")
            return

        # Nó de decisão: gera o if/else
        feat = nomes_features[feature[no_idx]]
        thr  = threshold[no_idx]
        linhas.append(f"{indent}if ({feat} <= {thr:.1f}) {{")
        gerar_no(left[no_idx],  profundidade + 1)
        linhas.append(f"{indent}}} else {{")
        gerar_no(right[no_idx], profundidade + 1)
        linhas.append(f"{indent}}}")

    gerar_no(0, 0)

    linhas.append("}")
    linhas.append("")
    linhas.append("#endif // MODELO_ARVORE_H")

    codigo = "\n".join(linhas)

    with open(caminho, 'w', encoding='utf-8') as f:
        f.write(codigo)

    print(f"\n✓ Código C++ exportado para '{caminho}'")
    print("\nPrévia do código gerado:")
    print("-" * 40)
    # Mostra as primeiras 30 linhas
    for linha in linhas[:30]:
        print(linha)
    print("  ...")

    return codigo


# =============================================================
# PARTE 5 — SALVAR O MODELO
#
# Salva o modelo treinado como arquivo .pkl (pickle).
# Permite carregar depois sem precisar retreinar.
# =============================================================

def salvar_modelo(modelo, caminho='modelo_seguidor.pkl'):
    with open(caminho, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"\n✓ Modelo salvo em '{caminho}'")
    print("  Para carregar depois: modelo = pickle.load(open('modelo_seguidor.pkl','rb'))")


# =============================================================
# EXECUÇÃO PRINCIPAL
# =============================================================

if __name__ == '__main__':
    print("="*50)
    print("PIPELINE COMPLETO — ÁRVORE DE DECISÃO")
    print("Seguidor de Linha com ESP32-S3")
    print("="*50 + "\n")

    # 1. Carregar ou criar dataset
    df = carregar_ou_criar_dataset('dados_pista.csv')

    # 2. Treinar o modelo
    modelo = treinar_modelo(df)

    # 3. Visualizar a árvore
    visualizar_arvore(modelo)

    # 4. Exportar para C++ (ESP32)
    exportar_para_cpp(modelo, 'modelo_arvore.h')

    # 5. Salvar o modelo
    salvar_modelo(modelo)

    print("\n" + "="*50)
    print("CONCLUÍDO!")
    print("Arquivos gerados:")
    print("  dados_pista.csv   → dataset usado no treino")
    print("  arvore_decisao.png → visualização da árvore")
    print("  modelo_arvore.h   → cole no projeto Arduino/ESP32")
    print("  modelo_seguidor.pkl → modelo salvo para retreinar")
    print("="*50)
