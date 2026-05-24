# 🏎️ Carrinho Seguidor de Linha com ESP32-S3 e Machine Learning

> Carrinho autônomo que usa **Árvore de Decisão (scikit-learn)** treinada no PC e exportada como C++ puro para o **ESP32-S3**. O modelo aprende a identificar trechos da pista (retas, curvas, cruzamentos, degraus) e ajusta automaticamente os parâmetros do **PID** para correr o mais rápido possível.

**🏁 Recorde atual da pista: 36 segundos — meta: bater esse tempo.**

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [A Pista](#-a-pista)
- [Hardware Utilizado](#-hardware-utilizado)
- [Como Funciona o Machine Learning](#-como-funciona-o-machine-learning)
- [Por que não usamos outros algoritmos?](#-por-que-não-usamos-outros-algoritmos)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Como Instalar e Usar](#-como-instalar-e-usar)
- [Fluxo de Treinamento e Retreino](#-fluxo-de-treinamento-e-retreino)
- [Parâmetros para Ajustar](#-parâmetros-para-ajustar)
- [Referências](#-referências)

---

## 📖 Sobre o Projeto

Este projeto foi desenvolvido como trabalho acadêmico com o objetivo de criar um carrinho seguidor de linha que utiliza **Machine Learning** para aprender o layout da pista e otimizar a velocidade em cada trecho.

A ideia central é separar duas etapas:

- **Treinamento** → feito no PC com Python (scikit-learn). Você coleta dados dos sensores, treina a Árvore de Decisão e exporta o modelo.
- **Execução** → feita no ESP32-S3 em tempo real. O modelo exportado vira código C++ puro — sem nenhuma biblioteca de ML no microcontrolador — tornando cada decisão extremamente rápida (menos de 2 µs).

O ciclo de retreino é o coração do projeto: cada volta na pista gera dados reais que tornam o modelo mais preciso, que torna o carrinho mais rápido.

---

## 🏁 A Pista

A pista utilizada neste projeto é feita com fita preta sobre fundo branco e possui quatro desafios principais:

| Trecho | Desafio |
|---|---|
| Reta longa | Área para acelerar ao máximo |
| Cruzamento em X | Risco de o carrinho escolher o caminho errado |
| Curva em S | Mudança de direção encadeada |
| Segmentos em degrau | Sequência de curvas rápidas e fechadas |

O cruzamento em X é o trecho mais crítico. Quando ambos os sensores detectam preto ao mesmo tempo, o carrinho ignora os sensores por 250 ms e segue reto — estratégia aprendida durante o retreino.

---

## 🔧 Hardware Utilizado

| Componente | Quantidade | Função |
|---|---|---|
| ESP32-S3 | 1 | Microcontrolador principal |
| Sensor TCRT5000 (IR) | 2 | Detectar a linha preta no chão |
| Ponte H L298N | 1 | Controlar a direção e velocidade dos motores |
| Motor DC com caixa de redução | 2 | Movimentar o carrinho |
| Chassi de 2 rodas | 1 | Estrutura física do carrinho |
| Bateria LiPo / pilhas | 1 | Alimentação |

### Por que ESP32-S3 em vez de Arduino UNO?

O Arduino UNO foi o ponto de partida do tutorial original da Eletrogate, mas foi substituído pelo ESP32-S3 por três motivos:

1. **Memória**: Arduino UNO tem 2 KB de RAM e 32 KB de Flash. O ESP32-S3 tem 512 KB de RAM e 8 MB de Flash — espaço suficiente para o modelo de ML.
2. **Processamento**: ESP32-S3 roda a 240 MHz dual-core, necessário para calcular o PID e executar a árvore em tempo real sem atraso.
3. **Ponto flutuante nativo**: o PID usa números decimais (float). No ESP32-S3 isso é feito em hardware; no Arduino UNO é emulado em software, muito mais lento.

### Por que os sensores em modo analógico?

Os sensores TCRT5000 podem operar em modo **digital** (retorna 0 ou 1) ou **analógico** (retorna 0 a 1023). Neste projeto usamos **modo analógico** porque:

- Modo digital → só 4 combinações possíveis (00, 01, 10, 11)
- Modo analógico → milhares de combinações → a Árvore aprende limiares exatos da sua pista

Isso permite distinguir, por exemplo, uma curva suave de uma curva fechada, e aplicar velocidades diferentes em cada caso.

---

## 🧠 Como Funciona o Machine Learning

### O algoritmo: DecisionTreeClassifier (scikit-learn)

A Árvore de Decisão lê os valores dos dois sensores e classifica cada momento da corrida em uma de cinco situações:

| Classe | Situação | Ação do carrinho |
|---|---|---|
| `0` — RETO | Linha centrada entre os dois sensores | Acelera ao máximo, PID suave |
| `1` — CURVA ESQ | Sensor esquerdo detecta a linha | PID agressivo, velocidade média |
| `2` — CURVA DIR | Sensor direito detecta a linha | PID agressivo, velocidade média |
| `3` — BUSCAR | Nenhum sensor detecta a linha | Gira devagar na última direção conhecida |
| `4` — CRUZAMENTO | Ambos os sensores detectam a linha | Ignora sensores por 250 ms, segue reto |

### PID Adaptativo

Com base na classificação da Árvore, o PID usa parâmetros diferentes por trecho:

```
Reta       → Kp = 0.8  velocidade = MÁXIMA   (vai rápido, correção suave)
Curva      → Kp = 1.8  velocidade = MÉDIA     (correção agressiva, mais devagar)
Buscar     → gira na última direção conhecida
Cruzamento → ignora sensores por 250 ms, segue reto
```

**Kp** é o ganho proporcional do PID — quanto maior, mais agressiva é a correção de rumo. Alto em curvas, baixo em retas. **Kd** amortece oscilações. **Ki** corrige erros acumulados (mantemos em zero para começar).

### Como o modelo é exportado para o ESP32

Após o treinamento no PC, as regras da árvore são convertidas automaticamente em uma função C++ simples:

```cpp
// Exemplo de código gerado automaticamente pelo Python
int predict(int s_esq, int s_dir) {
  if (s_esq <= 487.5) {
    if (s_dir <= 712.0) { return 3; }  // BUSCAR
    else                { return 2; }  // CURVA DIR
  } else {
    if (s_dir <= 712.0) { return 1; }  // CURVA ESQ
    else                { return 0; }  // RETO
  }
}
```

Essa função é copiada para o arquivo `modelo_arvore.h` e incluída diretamente no firmware do ESP32. Zero biblioteca de ML, zero overhead.

---

## ⚖️ Por que não usamos outros algoritmos?

Durante o projeto, avaliamos quatro abordagens antes de escolher a Árvore de Decisão:

### PID puro
Clássico e muito eficiente. Porém os parâmetros Kp, Ki, Kd são fixos para qualquer trecho da pista — o PID não sabe que está numa reta e poderia ir mais rápido, ou que está numa curva e precisa frear. Usamos o PID, mas combinado com a Árvore para que ele receba parâmetros diferentes por trecho.

### Q-Learning
Aprende durante a corrida na pista física, o que dificulta o treinamento offline no PC. Com apenas 2 sensores, o estado "ambos no preto" pode significar cruzamento ou linha centrada — ambiguidade que prejudica o aprendizado. Descartado.

### Random Forest (100 árvores)
Com 2 sensores IR, uma única árvore de profundidade 3 já captura todas as combinações possíveis. O Random Forest adicionaria cerca de 3 MB de Flash sem nenhum ganho real de precisão. Descartado.

| Algoritmo | Treina no PC? | Cabe no ESP32? | Ideal para 2 sensores? |
|---|---|---|---|
| **Árvore de Decisão** ⭐ | ✅ Sim | ✅ ~2 KB | ✅ Sim |
| PID puro | ❌ Calibração manual | ✅ ~50 bytes | ✅ Sim |
| Q-Learning | ⚠️ Parcial | ✅ ~48 bytes | ⚠️ Limitado |
| Random Forest (100 árvores) | ✅ Sim | ❌ ~3 MB | ❌ Overkill |

---

## 📁 Estrutura do Repositório

```
carrinho-ml-linha/
│
├── README.md                       ← Documentação completa do projeto
├── .gitignore
│
├── ml/
│   └── treinar_modelo.py           ← Script Python: treina e exporta o modelo
│
├── firmware/
│   ├── seguidor_linha_esp32.ino    ← Código principal do ESP32-S3
│   └── modelo_arvore.h             ← Gerado pelo Python (copie aqui após treinar)
│
└── dados/
    └── dados_pista.csv             ← Dataset de treino (sintético ou dados reais)
```

---

## 🚀 Como Instalar e Usar

### Pré-requisitos

**No PC — para treinar o modelo:**
```bash
pip install scikit-learn pandas numpy matplotlib
```

**No Arduino IDE — para programar o ESP32-S3:**
1. Vá em `File → Preferences → Additional Boards Manager URLs` e adicione:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
2. Vá em `Tools → Board → Boards Manager`, pesquise `esp32` e instale.
3. Selecione `Tools → Board → ESP32S3 Dev Module`.

---

### Passo 1 — Clone o repositório

```bash
git clone https://github.com/SEU_USUARIO/carrinho-ml-linha.git
cd carrinho-ml-linha
```

### Passo 2 — Treine o modelo no PC

```bash
cd ml
python treinar_modelo.py
```

Arquivos gerados automaticamente:

| Arquivo | Para que serve |
|---|---|
| `dados_pista.csv` | Dataset usado no treino — pode abrir e editar |
| `arvore_decisao.png` | Imagem visual da árvore com as regras aprendidas |
| `modelo_arvore.h` | **Cole este arquivo na pasta `firmware/`** |
| `modelo_seguidor.pkl` | Modelo salvo para retreinar depois |

### Passo 3 — Suba o firmware no ESP32-S3

1. Copie o `modelo_arvore.h` gerado para a pasta `firmware/`
2. Abra `firmware/seguidor_linha_esp32.ino` no Arduino IDE
3. Ajuste os pinos em `CONFIGURAÇÃO DE PINOS` conforme sua montagem
4. Selecione a porta em `Tools → Port`
5. Clique em **Upload**

### Passo 4 — Calibre antes da corrida

Abra o Monitor Serial (115200 baud) e posicione o carrinho sobre diferentes partes da pista:

```
S_ESQ=823  S_DIR=12   ACAO=1   ← funcionando corretamente
S_ESQ=45   S_DIR=891  ACAO=2
S_ESQ=498  S_DIR=511  ACAO=0
```

Valores esperados com TCRT5000:
- Sensor sobre linha preta → `800 a 1023`
- Sensor sobre fundo branco → `0 a 200`

---

## 🔄 Fluxo de Treinamento e Retreino

```
FASE 1 → Treinar com dados sintéticos no PC
         python treinar_modelo.py → gera modelo_arvore.h
              │
              ▼
FASE 2 → Subir no ESP32-S3
         Carrinho anda na pista com o modelo inicial
              │
              ▼
FASE 3 → Coletar dados reais
         Monitor Serial → copiar → salvar como dados_pista.csv
              │
              ▼
FASE 4 → Retreinar no PC  ◄── repete até bater o recorde
         python treinar_modelo.py (menos de 1 segundo)
         Novo modelo → subir no ESP32 → testar → coletar → retreinar
```

### Como coletar os dados reais

1. Abra o Monitor Serial no Arduino IDE (115200 baud)
2. Deixe o carrinho dar 5 a 10 voltas na pista
3. Copie todo o output do Monitor Serial
4. Salve como `dados/dados_pista.csv` com o cabeçalho `s_esq,s_dir,acao`
5. Rode `python treinar_modelo.py` novamente

> 💡 Esse ciclo é chamado de *active learning*: cada volta gera dados melhores → modelo mais preciso → carrinho mais rápido → dados ainda melhores.

### Documentando a evolução com commits

```bash
git add .
git commit -m "v1: treino sintético - primeira corrida"
git commit -m "v2: retreino com dados reais - tempo 33s"
git commit -m "v3: Kp_curva ajustado para 2.0 - tempo 31s"
git commit -m "v4: VEL_MAX aumentado para 210 - tempo 29s"
git push
```

---

## 🎛️ Parâmetros para Ajustar

```cpp
// Velocidades (0–255)
#define VEL_MAX    200   // retas → aumente para ir mais rápido
#define VEL_MEDIA  150   // curvas → reduza se perder a linha
#define VEL_BAIXA   90   // cruzamento

// PID para retas
float Kp_reta = 0.8;   // aumente se oscilar nas retas
float Kd_reta = 0.1;

// PID para curvas
float Kp_curva = 1.8;  // aumente se perder a linha nas curvas
float Kd_curva = 0.3;

// Cruzamento
int tempo_cruzamento_ms = 250;  // ms sem olhar os sensores
```

**Estratégia de ajuste:**
1. Comece com `VEL_MAX = 150` e aumente 10 pontos por vez
2. Oscilar nas retas → aumente `Kd_reta`
3. Perder a linha nas curvas → aumente `Kp_curva` ou reduza `VEL_MEDIA`
4. Problema no cruzamento → ajuste `tempo_cruzamento_ms`

---

## 📊 Histórico de Resultados

| Versão | Dataset | Tempo | Observação |
|---|---|---|---|
| v1.0 | Sintético | — | Primeira tentativa |
| v2.0 | Dados reais (1º retreino) | — | A medir |
| v3.0 | Dados reais (2º retreino) | — | A medir |
| **Meta** | — | **< 36s** | Recorde atual a bater |

---

## 📚 Referências

- [Tutorial Eletrogate — Robô Seguidor de Linha](https://blog.eletrogate.com/robo-seguidor-de-linha-tutorial-completo/)
- [scikit-learn — DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Lab seguidor de linha no Webots](https://felipenmartins.github.io/Robotics-Simulation-Labs/Lab2/)
- [Documentação ESP32-S3 — Espressif](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/)

---

*Última atualização: Maio 2026*
