import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
    plot_tree
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    accuracy_score
)

import pickle
import os


def criar_dataset_sintetico():

    print("Criando dataset sintético...")

    dados = []

    n = 400

    for _ in range(n):

        s_esq = np.random.randint(400, 750)
        s_dir = np.random.randint(400, 750)

        dados.append([
            s_esq,
            s_dir,
            0
        ])

    for _ in range(n):

        s_esq = np.random.randint(750, 1023)
        s_dir = np.random.randint(0, 250)

        dados.append([
            s_esq,
            s_dir,
            1
        ])

    for _ in range(n):

        s_esq = np.random.randint(0, 250)
        s_dir = np.random.randint(750, 1023)

        dados.append([
            s_esq,
            s_dir,
            2
        ])

    for _ in range(n):

        s_esq = np.random.randint(0, 200)
        s_dir = np.random.randint(0, 200)

        dados.append([
            s_esq,
            s_dir,
            3
        ])

    for _ in range(n):

        s_esq = np.random.randint(800, 1023)
        s_dir = np.random.randint(800, 1023)

        dados.append([
            s_esq,
            s_dir,
            4
        ])

    df = pd.DataFrame(
        dados,
        columns=[
            's_esq',
            's_dir',
            'acao'
        ]
    )

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print(f"Dataset criado: {len(df)} amostras")

    print(
        df['acao'].value_counts().rename({
            0: 'Reto',
            1: 'Curva Esq',
            2: 'Curva Dir',
            3: 'Buscar Linha',
            4: 'Cruzamento'
        })
    )

    return df


def carregar_ou_criar_dataset(
    caminho_csv='dados_pista.csv'
):

    if os.path.exists(caminho_csv):

        print(
            f"Carregando dados de '{caminho_csv}'..."
        )

        df = pd.read_csv(caminho_csv)

        print(
            f"Dados carregados: {len(df)} amostras"
        )

    else:

        print(
            f"Arquivo '{caminho_csv}' não encontrado."
        )

        df = criar_dataset_sintetico()

        df.to_csv(
            caminho_csv,
            index=False
        )

        print(
            f"Dataset salvo em '{caminho_csv}'"
        )

    return df


def treinar_modelo(df):

    print("\n" + "=" * 50)
    print("TREINANDO ÁRVORE")
    print("=" * 50)

    X = df[['s_esq', 's_dir']].values
    y = df['acao'].values

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"\nTreino: {len(X_treino)}"
    )

    print(
        f"Teste: {len(X_teste)}"
    )

    modelo = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        criterion='gini',
        random_state=42
    )

    modelo.fit(
        X_treino,
        y_treino
    )

    y_pred = modelo.predict(X_teste)

    acuracia = accuracy_score(
        y_teste,
        y_pred
    )

    print(
        f"\nAcurácia: {acuracia * 100:.1f}%"
    )

    print(
        classification_report(
            y_teste,
            y_pred,
            target_names=[
                'Reto',
                'CurvaEsq',
                'CurvaDir',
                'BuscarLinha',
                'Cruzamento'
            ]
        )
    )

    regras = export_text(
        modelo,
        feature_names=[
            's_esq',
            's_dir'
        ]
    )

    print("\nREGRAS:\n")
    print(regras)

    return modelo


def visualizar_arvore(modelo):

    fig, ax = plt.subplots(
        figsize=(20, 10)
    )

    plot_tree(
        modelo,
        feature_names=[
            'Sensor Esq',
            'Sensor Dir'
        ],
        class_names=[
            'Reto',
            'CurvaEsq',
            'CurvaDir',
            'Buscar',
            'Cruzamento'
        ],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )

    plt.title(
        'Árvore de Decisão',
        fontsize=14,
        pad=20
    )

    plt.tight_layout()

    plt.savefig(
        'arvore_decisao.png',
        dpi=150,
        bbox_inches='tight'
    )

    print(
        "\nImagem salva em 'arvore_decisao.png'"
    )

    plt.show()


def exportar_para_cpp(
    modelo,
    caminho='modelo_arvore.h'
):

    tree = modelo.tree_

    left = tree.children_left
    right = tree.children_right

    feature = tree.feature
    threshold = tree.threshold

    value = tree.value

    nomes_features = [
        's_esq',
        's_dir'
    ]

    nomes_acoes = [
        'RETO',
        'CURVA_ESQ',
        'CURVA_DIR',
        'BUSCAR_LINHA',
        'CRUZAMENTO'
    ]

    linhas = []

    linhas.append("#ifndef MODELO_ARVORE_H")
    linhas.append("#define MODELO_ARVORE_H")
    linhas.append("")

    linhas.append("#define ACAO_RETO 0")
    linhas.append("#define ACAO_CURVA_ESQ 1")
    linhas.append("#define ACAO_CURVA_DIR 2")
    linhas.append("#define ACAO_BUSCAR_LINHA 3")
    linhas.append("#define ACAO_CRUZAMENTO 4")

    linhas.append("")
    linhas.append(
        "int predict(int s_esq, int s_dir) {"
    )

    def gerar_no(no_idx, profundidade):

        indent = "  " * (
            profundidade + 1
        )

        if left[no_idx] == -1:

            classe = int(
                np.argmax(
                    value[no_idx][0]
                )
            )

            nome = nomes_acoes[classe]

            linhas.append(
                f"{indent}return {classe};"
            )

            return

        feat = nomes_features[
            feature[no_idx]
        ]

        thr = threshold[no_idx]

        linhas.append(
            f"{indent}if ({feat} <= {thr:.1f}) {{"
        )

        gerar_no(
            left[no_idx],
            profundidade + 1
        )

        linhas.append(
            f"{indent}}} else {{"
        )

        gerar_no(
            right[no_idx],
            profundidade + 1
        )

        linhas.append(
            f"{indent}}}"
        )

    gerar_no(0, 0)

    linhas.append("}")
    linhas.append("")
    linhas.append("#endif")

    codigo = "\n".join(linhas)

    with open(
        caminho,
        'w',
        encoding='utf-8'
    ) as f:

        f.write(codigo)

    print(
        f"\nCódigo exportado para '{caminho}'"
    )

    return codigo


def salvar_modelo(
    modelo,
    caminho='modelo_seguidor.pkl'
):

    with open(caminho, 'wb') as f:

        pickle.dump(modelo, f)

    print(
        f"\nModelo salvo em '{caminho}'"
    )


if __name__ == '__main__':

    print("=" * 50)
    print("PIPELINE ÁRVORE DE DECISÃO")
    print("=" * 50)

    df = carregar_ou_criar_dataset(
        'dados_pista.csv'
    )

    modelo = treinar_modelo(df)

    visualizar_arvore(modelo)

    exportar_para_cpp(
        modelo,
        'modelo_arvore.h'
    )

    salvar_modelo(modelo)

    print("\nCONCLUÍDO")