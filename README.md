Esse projeto tem como objeetivo realizar o agrupamento de dados mistos, categorizando como dados mistos os dados numéricos, categóricos e textuais.

A imagem abaixo representa o pipeline do projeto:

![image](https://github.com/user-attachments/assets/6a2f07e5-7ac9-49a2-b6cf-dfe87348a707)

A pasta start.py tem os códigos necessários para todos os procedimentos. A função distmix representa o código de Ahmad-Day para calcular dados numéricos e categóricos (esse código foi uma adpatação da biblioteca que existe em R), a função bert_txt serve para transformar a parte textual em embbeding utilizando o modelo BERT(Bidirectional Encoder Representations from Transformers), para utilizar o modelo ElMo(Embeddings from Language Model) peguei o código do github https://github.com/HIT-SCIR/ELMoForManyLangs, a função geraEmbedding é uma abstração pra facilitar a escolha de quando vai utilizar BERT e quando vai utilizar ElMo, a função criando_coluna_PCA serve para diminuir a dimensionalidade dos embbedings para conseguir trabalhar com eles.

Para rodar o projeto basta rodar um dos notebook, eu criei dois para organizar e facilitar com qual estava trabalhando, ambos tem os mesmos códigos.
