#ifndef MODELO_ARVORE_H
#define MODELO_ARVORE_H

#define ACAO_RETO           0
#define ACAO_CURVA_ESQ      1
#define ACAO_CURVA_DIR      2
#define ACAO_BUSCAR_LINHA   3
#define ACAO_CRUZAMENTO     4

int predict(int s_esq, int s_dir) {

  if (s_esq <= 325.0) {

    if (s_dir <= 474.5) {

      return 3;

    } else {

      return 2;
    }

  } else {

    if (s_dir <= 325.5) {

      return 1;

    } else {

      if (s_esq <= 774.0) {

        return 0;

      } else {

        return 4;
      }
    }
  }
}

#endif