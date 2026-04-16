# Open_loop — Plano de fases e prompts para implementação

## Objetivo final

Desenhar uma trajetória executável de temperatura de placa para a cryostage que, ao passar pelo modelo reduzido da cryostage e pelo solver 2D de congelamento, produza uma progressão aproximadamente linear da frente de congelamento ao longo do processo, e depois validar essa trajetória experimentalmente.

Este problema deve ser tratado como um problema de desenho **open-loop** de trajetória. Não é um problema de controlo fechado direto da frente de congelamento.

---

## Regra principal do projeto

Não abrir várias frentes ao mesmo tempo.

Só se avança para a fase seguinte quando a anterior estiver fechada.

Também não se deve voltar a criar excesso de documentação na raiz do repositório. O foco deve estar em:

- estabilizar o workflow ativo;
- fechar a definição do observable;
- só depois voltar a otimizar;
- e só no fim preparar a validação experimental e o paper.

---

## Estrutura global do plano

### Bloco A — estabilizar o problema

1. Fase 1 — alinhar defaults e workflow ativo
2. Fase 2 — fechar a definição operacional da frente
3. Fase 3 — testar robustez do observable e sensibilidade numérica

### Bloco B — estabilizar a pipeline ativa

4. Fase 4 — mini-refactor orientado ao workflow ativo

### Bloco C — otimizar e validar

5. Fase 5 — reimplementação limpa do estudo BO ativo
6. Fase 6 — congelar o default operacional e preparar validação experimental
7. Fase 7 — fechar o pacote científico do artigo

---

# Fase 1 — Fixar o workflow ativo e limpar defaults históricos

## Para que serve

Garantir que o código que vai ser usado corresponde ao workflow atual do projeto, e não a defaults históricos herdados de estudos antigos.

## O que deve sair desta fase

- runner principal alinhado com o workflow atual;
- defaults coerentes com o workflow ativo pretendido, sem ancorar a discussão num knot count específico;
- scripts legacy preservados, mas claramente separados do caminho ativo.

## Esta fase fecha quando

- `run_open_loop_optimization.py` deixa de arrancar por defeito com settings históricos herdados;
- o schedule ativo fica explícito;
- o workflow ativo deixa de estar ancorado a um knot count específico.

## Prompt da Fase 1

```text
You are working inside the code_simulation folder of the Open_loop repository.

Task: align the active runnable workflow with the intended active operational workflow, without redesigning the repository and without changing the solver physics.

Goals:
1. Make the active main runner reflect the intended active workflow.
2. Remove historical default confusion from the active path.
3. Preserve legacy scripts, but make sure they are not mistaken for the active workflow.

Requirements:
- In run_open_loop_optimization.py, make the active defaults explicit.
- In open_loop_workflow_config.py, make the active workflow defaults explicit without anchoring the active path to a specific knot count.
- Keep historical support for older parameterizations, but do not use them as the active default.
- Do not change solver physics, objective logic, or BO logic in this phase.
- Do not create many new markdown files.

At the end, provide:
- files changed
- exact active defaults after the patch
- what legacy behavior was preserved
```

---

# Fase 2 — Fechar a definição operacional da frente de congelamento

## Para que serve

Fechar a definição do observable principal antes de retomar qualquer estudo BO sério.

Como o solver usa uma abordagem de calor latente por capacidade calorífica aparente com `dT_mushy`, a frente de congelamento não é uma interface matemática nítida. Portanto, `z_front(t)` tem de passar a ser uma definição explícita, controlada e auditável.

## O que deve sair desta fase

- definição explícita da frente no código;
- seleção clara de modo de definição da frente;
- output da run a registar qual definição foi usada.

## Modos mínimos a considerar

- `isotherm_Tf`
- `solidus`
- `freeze_threshold`

## Esta fase fecha quando

- a definição da frente já não está hardcoded de forma implícita;
- o output da run regista claramente a definição usada;
- o critério de paragem por congelamento completo ainda pode manter-se igual por agora.

## Prompt da Fase 2

```text
You are working inside the code_simulation folder of the Open_loop repository.

Task: make the freezing-front definition explicit and selectable in the active workflow, without yet changing the freeze-complete stopping logic.

Goals:
1. Remove implicit front-definition ambiguity.
2. Allow the front observable to be extracted from an explicit threshold rule.
3. Record the chosen front-definition mode in the outputs.

Requirements:
- In front_tracking.py, add a generic threshold-based front extraction function.
- Keep backward compatibility where possible.
- In solver.py, add an explicit front_definition_mode argument with at least:
  - isotherm_Tf
  - solidus
  - freeze_threshold
- Map each mode to a temperature threshold consistently.
- Use that threshold for centerline, wall, and curved-front extraction.
- Record the selected front_definition_mode and threshold in the run output or summary.
- Do not change the freeze-complete stopping criterion in this phase.
- Do not run broad studies.

At the end, provide:
- files changed
- front-definition modes implemented
- where the selected mode is recorded in outputs
```

---

# Fase 3 — Diagnóstico curto de observáveis e sensibilidade numérica

## Para que serve

Antes de otimizar, é preciso perceber se o observable que vai ser usado é robusto.

Nesta fase não se procura ainda a trajetória ótima. Procura-se perceber:

- se `z_front(t)` muda muito com a definição da frente;
- se `dT_mushy` altera materialmente os resultados;
- se faz sentido usar também front-passage times nas alturas dos termopares.

## O que deve sair desta fase

- comparação curta entre definições de frente;
- análise curta de sensibilidade a `dT_mushy`;
- recomendação operacional sobre qual observable deve entrar no objetivo BO.

## Esta fase fecha quando

- o observable a usar no BO está decidido;
- está claro se o objetivo vai usar tracking de posição da frente, front-passage times, ou uma combinação dos dois.

## Prompt da Fase 3

```text
You are working inside the code_simulation folder of the Open_loop repository.

Task: run a narrow scientific diagnostic of the front observable before any new BO campaign.

Goals:
1. Compare the active front observable across front-definition modes.
2. Assess sensitivity to dT_mushy.
3. Decide which observable is robust enough for optimization.

Requirements:
- Use only a small number of representative baseline cases.
- Compare at least:
  - isotherm_Tf
  - solidus
  - freeze_threshold
- Evaluate sensitivity to at least two or three dT_mushy values.
- Extract and compare:
  - z_front(t)
  - front-passage times at the probe heights
  - any relevant stability/noise indicators
- Do not run any new BO study in this phase.
- Produce a concise result summary focused only on observable robustness and recommendation.

At the end, provide:
- cases tested
- observable comparison summary
- recommended observable for the BO objective
- whether the current front-position tracking remains acceptable
```

---

# Fase 4 — Reestruturar minimamente a pipeline ativa

## Para que serve

Depois de estabilizar o problema científico, faz-se uma reestruturação pequena e controlada para deixar a pipeline ativa clara.

A cadeia que se pretende cristalina é:

`theta -> T_ref(t) -> T_plate(t) -> solver -> observable -> J(theta)`

## O que deve sair desta fase

Separação clara entre:

- config;
- cascade executável;
- objective/evaluation;
- backend BO;
- orchestration/logging.

## Esta fase fecha quando

- `open_loop_problem.py` trata da formulação do problema e do objetivo;
- `open_loop_cascade.py` trata apenas da cascade executável;
- `open_loop_bayesian_optimizer.py` trata apenas do backend BO;
- `open_loop_optimizer.py` trata da orquestração e logging.

## Prompt da Fase 4

```text
You are working inside the code_simulation folder of the Open_loop repository.

Task: perform a minimal workflow-oriented refactor of the active open-loop pipeline, without redesigning the whole repository.

Goals:
1. Clarify the active chain:
   theta -> T_ref(t) -> T_plate(t) -> freezing solver -> observable -> objective
2. Separate responsibilities across the existing active files.
3. Keep legacy scripts intact but outside the active conceptual path.

Requirements:
- Keep file moves to a minimum.
- Clarify responsibilities of:
  - open_loop_workflow_config.py
  - open_loop_cascade.py
  - open_loop_problem.py
  - open_loop_bayesian_optimizer.py
  - open_loop_optimizer.py
- Do not redesign the whole codebase.
- Do not change the scientific objective in this phase.
- Do not run large studies.

At the end, provide:
- files changed
- responsibility map after the refactor
- what was deliberately left untouched
```

---

# Fase 5 — Reimplementação limpa do estudo BO ativo

## Para que serve

Retomar o BO só depois de o workflow estar estabilizado.

Mas esta fase deve ser estreita e predefinida, não uma exploração larga.

## Opções de primeira comparação

### Caminho A — comparação entre parametrizações de trajetória

- manter o mesmo observable já fechado na Fase 3;
- comparar duas parametrizações candidatas da trajetória sob o mesmo workflow ativo.

### Caminho B — comparação entre parametrizações temporais

- manter a mesma família geral de trajetória;
- comparar duas representações temporais sob as mesmas regras de admissibilidade e o mesmo objetivo.

Não fazer os dois caminhos ao mesmo tempo.

## O que deve sair desta fase

- runner BO limpo;
- settings fixos e rastreáveis;
- comparação curta e reprodutível.

## Esta fase fecha quando

- existe uma run limpa e reprodutível;
- já não há defaults escondidos;
- a pergunta BO está bem delimitada.

## Prompt da Fase 5

```text
You are working inside the code_simulation folder of the Open_loop repository.

Task: implement a clean, narrow BO study on top of the stabilized active workflow.

Scientific scope:
Do not run a broad exploration. Run only one pre-defined narrow study.

Goals:
1. Use the active stabilized observable and objective.
2. Use the active admissibility rules.
3. Produce a reproducible BO comparison with fixed settings.

Requirements:
- Use the active default external schedule family unless explicitly overridden.
- Fix the BO budget and seed policy in advance.
- Use a narrow comparison only between a small number of candidate trajectory parameterizations.
- Keep the study small and interpretable.
- Report:
  - best objective
  - stability across seeds
  - feasible vs rejected evaluations
  - resulting T_ref(t), T_plate(t), and front trajectory
- Do not open a broad knot-count sweep.
- Do not create many new memo files.

At the end, provide:
- exact study settings
- summary of results
- recommendation on whether to carry forward one operational default
```

---

# Fase 6 — Congelar o default operacional e preparar validação experimental

## Para que serve

Depois do estudo BO limpo, é preciso escolher uma trajetória para avançar e prepará-la para execução real na cryostage.

## O que deve sair desta fase

- uma trajetória escolhida;
- formato simples para exportar a trajetória;
- protocolo experimental curto para baseline vs otimizada.

## Esta fase fecha quando

- existe uma trajetória final selecionada para teste;
- está claro como comparar baseline vs otimizada;
- estão definidas as métricas experimentais principais.

## Prompt da Fase 6

```text
You are working inside the Open_loop repository.

Task: freeze one operational carry-forward trajectory and prepare it for experimental validation on the real cryostage.

Goals:
1. Select one trajectory to carry forward.
2. Export it in a form suitable for experimental execution.
3. Define a minimal baseline-versus-optimized validation protocol.

Requirements:
- Use the stabilized BO result from the active workflow.
- Export the selected reference trajectory in a simple executable format.
- Define the experimental comparison against baseline protocols.
- Use observables that can be inferred experimentally from the probe data.
- Keep the protocol concise and realistic.
- Do not start writing the paper in this phase.

At the end, provide:
- selected carry-forward trajectory
- export format
- proposed validation protocol
- metrics to compare baseline vs optimized runs
```

---

# Fase 7 — Fechar o pacote científico do artigo

## Para que serve

Transformar o workflow estabilizado e validado num pacote de artigo.

A narrativa deve ficar centrada em:

- validação do framework em protocolos de temperatura constante;
- comparação baseline vs otimizada em simulação;
- execução experimental de trajetórias otimizadas;
- melhoria da regulação da frente de congelamento.

Não deve ficar centrada em provar uma parametrização de trajetória globalmente ótima.

## O que deve sair desta fase

- conjunto final de figuras e tabelas;
- narrativa coerente do paper;
- claims conservadoras e alinhadas com a evidência.

## Esta fase fecha quando

- os outputs principais do paper estão organizados;
- a conclusão está centrada na regulação da frente e na demonstração experimental;
- questões metodológicas ainda em aberto continuam explicitamente tratadas como limitações.

## Prompt da Fase 7

```text
You are working inside the Open_loop repository.

Task: prepare the final article-facing package from the validated active workflow and experimental results.

Goals:
1. Organize results around the paper narrative.
2. Avoid overclaiming unresolved optimization details.
3. Focus the paper on freezing-front regulation and experimental demonstration.

Requirements:
- Structure outputs around:
  - validation under constant-temperature protocols
  - simulation comparison of baseline vs optimized trajectories
  - experimental execution of optimized trajectories
- Do not frame the paper as proof of a globally optimal trajectory parameterization.
- Keep cryostage characterization details secondary or supplementary.
- Identify the final figure and table set needed for the paper.
- Keep the scientific claims conservative and aligned with the current evidence.

At the end, provide:
- proposed final figure list
- proposed final table list
- concise claim set for the paper
- unresolved points that should remain explicitly framed as limitations
```

---

# Resumo operacional final

## Ordem recomendada

1. Fase 1 — defaults e workflow ativo
2. Fase 2 — definição da frente
3. Fase 3 — robustez do observable
4. Fase 4 — mini-refactor da pipeline ativa
5. Fase 5 — BO estreito e limpo
6. Fase 6 — validação experimental
7. Fase 7 — pacote do paper

## Recomendação prática imediata

O melhor ponto para começar é:

- Fase 1;
- depois Fase 2;
- e parar aí para rever antes de avançar.

O maior risco neste momento não é falta de BO. O maior risco é voltar a otimizar antes de fechar bem a definição do observable da frente.
