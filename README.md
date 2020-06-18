# Local Binary Pattern (liveness detection)

## Modo de usar

### Inputs

Na pasta videos_reais voce ira colocar a juncao em apenas um unico video.mp4, todos os videos gravados que tenham pessoas reais.

Na pasta videos_fakes voce ira colocar a juncao em apenas um unico video.mp4, todos os videos gravados que sejam tentativas de burlar o
face recognition (videos de pessoas atraves do celular).

### gen_frames

Voce ira rodar o arquivo gen_frames.py, ele ira gerar os frames dos videos e grava los nas pastas frames_reais e frames_fakes

### gen_pickle

Voce ira rodar o arquivo gen_pickle.py, ele ira extrair o Local Binary Pattern dos frames (variaveis preditoras) e seus rotulos
(real, fake: variavel resposta) e ira gravalos na pasta dados.

### gen_models

Voce ira rodar o arquivo gen_models.py, ele ira gerar os modelos do tipo random forest,knn,logistico e svc linear. Os algoritmos
ja passaram pelo ajuste dos hiperparametros usando grid search, porem sinta se livre para muda los. Esse codigo tambem ira trazer
as matrizes de confusao

### real_time

Voce ira rodar o arquivo real_time.py, ele ira abrir sua webcam para testar em real time se o liveness detection esta funcionando

