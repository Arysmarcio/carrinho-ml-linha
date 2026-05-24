// =============================================================
//  SEGUIDOR DE LINHA — ESP32-S3
//  Árvore de Decisão (classificação de trecho) + PID adaptativo
//
//  Hardware necessário (conforme site Eletrogate):
//    - ESP32-S3
//    - 2x Sensor TCRT5000 (leitura ANALÓGICA)
//    - Ponte H L298N
//    - 2x Motores DC
//
//  Como usar:
//    1. Cole o arquivo modelo_arvore.h nesta mesma pasta
//    2. Ajuste os pinos em "CONFIGURAÇÃO DE PINOS"
//    3. Ajuste VELOCIDADE_BASE e os perfis de PID
//    4. Faça upload para o ESP32-S3
// =============================================================

#include "modelo_arvore.h"   // ← arquivo gerado pelo Python

// =============================================================
// CONFIGURAÇÃO DE PINOS
// Ajuste conforme a sua montagem física
// =============================================================

// Sensores TCRT5000 — use pinos ADC do ESP32-S3
// No ESP32-S3 os pinos ADC válidos são: 1-10 (GPIO1 a GPIO10)
#define PINO_SENSOR_ESQ  4   // GPIO4 → sensor esquerdo
#define PINO_SENSOR_DIR  5   // GPIO5 → sensor direito

// Ponte H L298N — controle dos motores
#define MOTOR_ESQ_IN1   16   // GPIO16 → motor esquerdo direção A
#define MOTOR_ESQ_IN2   17   // GPIO17 → motor esquerdo direção B
#define MOTOR_ESQ_EN    18   // GPIO18 → motor esquerdo velocidade (PWM)

#define MOTOR_DIR_IN1   19   // GPIO19 → motor direito direção A
#define MOTOR_DIR_IN2   20   // GPIO20 → motor direito direção B
#define MOTOR_DIR_EN    21   // GPIO21 → motor direito velocidade (PWM)

// =============================================================
// CONFIGURAÇÃO PWM (ESP32-S3 usa ledc)
// =============================================================
#define PWM_FREQ      1000   // frequência do PWM em Hz
#define PWM_RES          8   // resolução: 8 bits = 0 a 255
#define CH_ESQ           0   // canal PWM motor esquerdo
#define CH_DIR           1   // canal PWM motor direito

// =============================================================
// PARÂMETROS DE VELOCIDADE E PID
//
// Explicação do PID adaptativo:
//   Dependendo do trecho da pista (classificado pela Árvore),
//   usamos parâmetros diferentes de velocidade e PID.
//
//   Kp = ganho proporcional → quanto mais alto, mais agressivo
//        é a correção de rumo. Alto em curvas, baixo em retas.
//   Ki = ganho integral → corrige erro acumulado ao longo do tempo.
//        Mantenha baixo (ou zero) para começar.
//   Kd = ganho derivativo → amortece oscilações.
//        Ajuda em curvas fechadas.
//
//   VELOCIDADE_BASE = velocidade dos dois motores quando no centro.
//   Reduza nas curvas para não sair da linha.
// =============================================================

// Velocidade máxima dos motores (0–255)
#define VEL_MAX    200
#define VEL_MEDIA  150
#define VEL_BAIXA   90
#define VEL_BUSCA   70

// Perfil RETA LONGA: vai rápido, PID suave
float Kp_reta = 0.8;
float Ki_reta = 0.0;
float Kd_reta = 0.1;
int   vel_reta = VEL_MAX;

// Perfil CURVA: velocidade média, PID mais agressivo
float Kp_curva = 1.8;
float Ki_curva = 0.0;
float Kd_curva = 0.3;
int   vel_curva = VEL_MEDIA;

// Perfil CRUZAMENTO: ignora sensores por tempo fixo, segue reto
int   vel_cruzamento = VEL_BAIXA;
int   tempo_cruzamento_ms = 250;   // ms para atravessar o cruzamento

// Perfil BUSCA (perdeu a linha): gira devagar
int   vel_busca = VEL_BUSCA;

// =============================================================
// VARIÁVEIS GLOBAIS DO PID
// =============================================================
float erro_anterior = 0.0;
float integral      = 0.0;
unsigned long t_anterior = 0;

// Controle do cruzamento
bool  em_cruzamento = false;
unsigned long t_cruzamento = 0;

// =============================================================
// FUNÇÕES DE CONTROLE DOS MOTORES
// =============================================================

void configurar_motores() {
  // Configura direção
  pinMode(MOTOR_ESQ_IN1, OUTPUT);
  pinMode(MOTOR_ESQ_IN2, OUTPUT);
  pinMode(MOTOR_DIR_IN1, OUTPUT);
  pinMode(MOTOR_DIR_IN2, OUTPUT);

  // Configura PWM (ESP32-S3 usa ledcSetup + ledcAttachPin)
  ledcSetup(CH_ESQ, PWM_FREQ, PWM_RES);
  ledcSetup(CH_DIR, PWM_FREQ, PWM_RES);
  ledcAttachPin(MOTOR_ESQ_EN, CH_ESQ);
  ledcAttachPin(MOTOR_DIR_EN, CH_DIR);
}

// vel_esq e vel_dir: -255 a 255
// valores negativos = ré
void mover(int vel_esq, int vel_dir) {
  // Motor esquerdo
  if (vel_esq >= 0) {
    digitalWrite(MOTOR_ESQ_IN1, HIGH);
    digitalWrite(MOTOR_ESQ_IN2, LOW);
    ledcWrite(CH_ESQ, constrain(vel_esq, 0, 255));
  } else {
    digitalWrite(MOTOR_ESQ_IN1, LOW);
    digitalWrite(MOTOR_ESQ_IN2, HIGH);
    ledcWrite(CH_ESQ, constrain(-vel_esq, 0, 255));
  }

  // Motor direito
  if (vel_dir >= 0) {
    digitalWrite(MOTOR_DIR_IN1, HIGH);
    digitalWrite(MOTOR_DIR_IN2, LOW);
    ledcWrite(CH_DIR, constrain(vel_dir, 0, 255));
  } else {
    digitalWrite(MOTOR_DIR_IN1, LOW);
    digitalWrite(MOTOR_DIR_IN2, HIGH);
    ledcWrite(CH_DIR, constrain(-vel_dir, 0, 255));
  }
}

void parar() {
  ledcWrite(CH_ESQ, 0);
  ledcWrite(CH_DIR, 0);
}

// =============================================================
// CÁLCULO DO PID
//
// Entrada: erro atual (diferença entre sensores)
//          Kp, Ki, Kd escolhidos pela Árvore de Decisão
// Saída:   correção a aplicar nos motores
// =============================================================

float calcular_pid(float erro, float Kp, float Ki, float Kd) {
  unsigned long t_atual = millis();
  float dt = (t_atual - t_anterior) / 1000.0;   // tempo em segundos
  if (dt <= 0) dt = 0.001;                        // evita divisão por zero

  // Proporcional: reage ao erro atual
  float P = Kp * erro;

  // Integral: soma erros ao longo do tempo (corrige desvio persistente)
  integral += erro * dt;
  integral  = constrain(integral, -500, 500);     // limita windup
  float I   = Ki * integral;

  // Derivativo: reage à velocidade de mudança do erro
  float D = Kd * (erro - erro_anterior) / dt;

  erro_anterior = erro;
  t_anterior    = t_atual;

  return P + I + D;
}

// =============================================================
// SETUP
// =============================================================

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("Seguidor de Linha - ESP32-S3");
  Serial.println("Árvore de Decisão + PID Adaptativo");
  Serial.println("====================================");

  // Configura pinos dos sensores como entrada analógica
  // No ESP32-S3 os pinos ADC não precisam de pinMode
  analogReadResolution(10);     // resolução de 10 bits = 0 a 1023
  analogSetAttenuation(ADC_11db); // range 0–3.3V

  configurar_motores();

  t_anterior = millis();
  Serial.println("Iniciando em 2 segundos...");
  delay(2000);
}

// =============================================================
// LOOP PRINCIPAL
//
// Fluxo de execução a cada iteração:
//   1. Lê os sensores analógicos
//   2. Usa a Árvore de Decisão para classificar o trecho
//   3. Seleciona parâmetros de PID e velocidade para o trecho
//   4. Calcula o erro e aplica o PID
//   5. Envia velocidade corrigida para os motores
// =============================================================

void loop() {

  // ----------------------------------------------------------
  // PASSO 1: Ler os sensores
  // analogRead retorna 0–1023 (10 bits de resolução)
  // Valor alto (~800+) = sensor sobre linha preta
  // Valor baixo (~0–200) = sensor sobre fundo branco
  // ----------------------------------------------------------
  int s_esq = analogRead(PINO_SENSOR_ESQ);
  int s_dir = analogRead(PINO_SENSOR_DIR);

  // ----------------------------------------------------------
  // PASSO 2: Classificar o trecho com a Árvore de Decisão
  // A função predict() foi gerada automaticamente pelo Python
  // e está no arquivo modelo_arvore.h
  // ----------------------------------------------------------
  int acao = predict(s_esq, s_dir);

  // Debug: mostra leituras e ação no Monitor Serial
  Serial.printf("S_ESQ=%4d  S_DIR=%4d  ACAO=%d\n", s_esq, s_dir, acao);

  // ----------------------------------------------------------
  // PASSO 3: Calcular o erro para o PID
  //
  // Erro = diferença normalizada entre os sensores (-1 a +1)
  //   0    = linha perfeitamente centrada
  //   +1   = totalmente à direita
  //   -1   = totalmente à esquerda
  //
  // Normalizamos dividindo pelo valor máximo (1023)
  // para que o erro não dependa da escala dos sensores.
  // ----------------------------------------------------------
  float erro = (float)(s_dir - s_esq) / 1023.0;

  // ----------------------------------------------------------
  // PASSO 4: Executar ação baseada na classificação da Árvore
  // ----------------------------------------------------------

  switch (acao) {

    // ── RETO: alta velocidade, PID suave ─────────────────────
    case ACAO_RETO: {
      integral = 0;   // reseta integral ao entrar numa reta
      float correcao = calcular_pid(erro, Kp_reta, Ki_reta, Kd_reta);
      int v_esq = vel_reta - (int)correcao;
      int v_dir = vel_reta + (int)correcao;
      mover(constrain(v_esq, 0, 255), constrain(v_dir, 0, 255));
      break;
    }

    // ── CURVA ESQUERDA: velocidade média, PID agressivo ──────
    case ACAO_CURVA_ESQ: {
      float correcao = calcular_pid(erro, Kp_curva, Ki_curva, Kd_curva);
      int v_esq = vel_curva - (int)correcao;
      int v_dir = vel_curva + (int)correcao;
      mover(constrain(v_esq, 0, 255), constrain(v_dir, 0, 255));
      break;
    }

    // ── CURVA DIREITA: igual curva esquerda (PID simétrico) ──
    case ACAO_CURVA_DIR: {
      float correcao = calcular_pid(erro, Kp_curva, Ki_curva, Kd_curva);
      int v_esq = vel_curva - (int)correcao;
      int v_dir = vel_curva + (int)correcao;
      mover(constrain(v_esq, 0, 255), constrain(v_dir, 0, 255));
      break;
    }

    // ── CRUZAMENTO: ignora sensores, segue reto por 250ms ────
    case ACAO_CRUZAMENTO: {
      if (!em_cruzamento) {
        em_cruzamento = true;
        t_cruzamento  = millis();
        integral = 0;   // reseta integral
        Serial.println(">>> CRUZAMENTO DETECTADO — seguindo reto");
      }
      if (millis() - t_cruzamento < tempo_cruzamento_ms) {
        // Durante o cruzamento: segue reto sem PID
        mover(vel_cruzamento, vel_cruzamento);
      } else {
        // Saiu do cruzamento
        em_cruzamento = false;
        Serial.println(">>> Saiu do cruzamento");
      }
      break;
    }

    // ── BUSCAR LINHA: gira até encontrar a linha ─────────────
    case ACAO_BUSCAR_LINHA: {
      integral = 0;
      // Gira em torno de si mesmo baseado no último erro conhecido
      if (erro_anterior >= 0) {
        mover(vel_busca, -vel_busca);   // gira para a direita
      } else {
        mover(-vel_busca, vel_busca);   // gira para a esquerda
      }
      break;
    }

    default:
      parar();
      break;
  }

  // Pequena pausa para estabilizar leituras ADC
  delayMicroseconds(500);
}
