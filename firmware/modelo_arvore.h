// ============================================================
// MODELO DE ÁRVORE DE DECISÃO — GERADO AUTOMATICAMENTE
// Treinado com scikit-learn DecisionTreeClassifier
// Cole este arquivo na pasta do seu projeto Arduino/ESP32
// ============================================================

#ifndef MODELO_ARVORE_H
#define MODELO_ARVORE_H

// Ações possíveis
#define ACAO_RETO         0
#define ACAO_CURVA_ESQ    1
#define ACAO_CURVA_DIR    2
#define ACAO_BUSCAR_LINHA 3
#define ACAO_CRUZAMENTO   4

// Função gerada pela árvore de decisão treinada
// s_esq: leitura analógica do sensor esquerdo (0–1023)
// s_dir: leitura analógica do sensor direito  (0–1023)
// retorna: código da ação a executar
int predict(int s_esq, int s_dir) {
  if (s_esq <= 325.0) {
    if (s_dir <= 474.5) {
      return 3; // BUSCAR_LINHA
    } else {
      return 2; // CURVA_DIR
    }
  } else {
    if (s_dir <= 325.5) {
      return 1; // CURVA_ESQ
    } else {
      if (s_esq <= 774.0) {
        return 0; // RETO
      } else {
        return 4; // CRUZAMENTO
      }
    }
  }
}

#endif // MODELO_ARVORE_H