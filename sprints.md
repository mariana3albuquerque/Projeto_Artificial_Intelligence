=Smartphone-Based Skin Lesion Classification for Early Triage Support
1. Título do projeto
Smartphone-Based Skin Lesion Classification for Early Triage Support
O projeto propõe o desenvolvimento de uma solução de apoio à triagem inicial de lesões de pele por meio de inteligência artificial aplicada a imagens médicas. O objetivo é identificar casos suspeitos e auxiliar a priorização do encaminhamento clínico. A proposta não pretende substituir o especialista, mas funcionar como ferramenta de apoio à decisão.
2. Participantes
•
Luiza Ehrenberger
•
Mariana Albuquerque
3. Divisão de responsabilidades entre os participantes
A divisão de responsabilidades foi definida para distribuir as atividades técnicas, analíticas e documentais do projeto.
Luiza Ehrenberger
Responsável pela parte técnica do projeto:
•
organização do pipeline de implementação;
•
preparação e pré-processamento da base de dados;
•
treinamento do modelo de classificação;
•
avaliação quantitativa dos resultados;
•
desenvolvimento do protótipo em formato web app.
Mariana Albuquerque
Responsável pela parte analítica e documental:
•
revisão bibliográfica e fundamentação do problema clínico;
•
documentação da base de dados e das classes analisadas;
•
análise qualitativa dos resultados;
•
apoio à interpretação das saídas do modelo;
•
redação do relatório final e organização da apresentação.
Responsabilidades compartilhadas
As seguintes atividades serão realizadas em conjunto:
•
definição do problema clínico;
•
escolha e justificativa da base de dados;
•
definição das métricas de avaliação;
•
discussão dos resultados;
•
revisão final do trabalho;
•
preparação da apresentação.
Essa organização está de acordo com a proposta da disciplina, que exige a definição do problema, do dataset, da métrica clínica e do impacto da solução no fluxo do profissional de saúde.
4. Lista de tarefas conforme o cronograma de sprints
O cronograma foi dividido em sprints curtos para tornar a implementação viável dentro do tempo da disciplina. A proposta é desenvolver uma prova de conceito funcional, e não um sistema clínico completo.
Sprint 1 — Organização da base e preparação dos dados
Objetivo: preparar a base de dados e organizar o ambiente do projeto.
Tarefas:
•
revisar a proposta final;
•
organizar a base HAM10000;
•
documentar as sete classes do dataset;
•
analisar a distribuição das classes;
•
definir treino, validação e teste;
•
implementar o pipeline inicial de carregamento e pré-processamento;
•
configurar o ambiente de desenvolvimento.
Sprint 2 — Desenvolvimento do modelo baseline
Objetivo: implementar um primeiro modelo simples de classificação.
Tarefas:
•
construir um modelo baseline de classificação de imagens;
•
treinar a primeira versão do classificador;
•
avaliar o desempenho com recall, precision, F1-score, ROC-AUC e matriz de confusão;
•
registrar os resultados iniciais.
Sprint 3 — Melhorias simples no modelo
Objetivo: melhorar o desempenho do modelo sem aumentar excessivamente a complexidade da implementação.
Tarefas:
•
aplicar data augmentation;
•
testar ajustes simples de treinamento;
•
comparar os resultados com o baseline;
•
priorizar a melhora da sensibilidade para lesões suspeitas.
Sprint 4 — Desenvolvimento do protótipo web
Objetivo: construir uma interface simples para demonstrar o funcionamento do sistema.
Tarefas:
•
desenvolver uma interface web básica para upload de imagens;
•
integrar o modelo treinado ao sistema;
•
exibir a classe predita e a probabilidade associada;
•
incluir um aviso de que a ferramenta serve apenas como apoio à triagem.
Sprint 5 — Testes finais e entrega
Objetivo: finalizar o projeto e preparar a entrega.
Tarefas:
•
realizar testes finais do modelo e do protótipo;
•
consolidar as métricas finais;
•
redigir a versão final do relatório;
•
preparar a apresentação;
•
revisar as limitações do projeto e possíveis melhorias futuras.
Para manter o projeto simples e viável, o foco principal será:
•
organizar a base HAM10000;
•
treinar um modelo baseline funcional;
•
apresentar recall, precision, F1-score e matriz de confusão;
•
construir um protótipo web simples com upload de imagem e predição;
•
concluir o relatório e a apresentação.
Esse cronograma está alinhado à proposta da disciplina, mas foi simplificado para priorizar uma implementação realista.
5. Trabalho executado até esse momento
Até o momento, o grupo já realizou as seguintes atividades:
•
definição do tema do projeto na área de dermatologia, com foco na classificação de lesões de pele para apoio à triagem inicial;
•
definição do problema clínico, considerando a importância da detecção precoce de lesões malignas;
•
elaboração e entrega da proposta final do projeto;
•
definição do título do projeto e das participantes;
•
escolha da base de dados principal, com ênfase no HAM10000, utilizado como benchmark inicial para desenvolvimento e avaliação;
•
definição da métrica clínica prioritária, estabelecendo a sensibilidade (recall) como principal indicador;
•
definição das métricas complementares, incluindo precision, F1-score, ROC-AUC e matriz de confusão;
•
organização inicial do cronograma de implementação;
•
definição do formato de implementação após o feedback do professor, estabelecendo que a solução será desenvolvida inicialmente como protótipo web app, e não como aplicativo nativo;
•
delimitação do escopo do projeto para que a implementação permaneça simples e viável dentro do tempo da disciplina.
Esses pontos estão de acordo com a proposta final, que descreve o problema clínico, as participantes, a base HAM10000 e a prioridade de sensibilidade para lesões suspeitas. A proposta também reconhece que existe uma diferença entre o cenário de uso desejado, baseado em smartphone, e a base utilizada, composta majoritariamente por imagens dermatoscópicas. Por isso, o projeto é tratado como uma prova de conceito, e não como um sistema pronto para uso clínico real.