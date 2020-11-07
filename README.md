# balanced_ensemble_credit_scoring
Este trabalho consiste em utilizar técnicas de balanceamento de dados baseados emsenble para classificar bases de crédito com diferentes níveis de desbalanceamento entre classes.

## Instalação
### Instalação das base [aqui](https://drive.google.com/file/d/12B45siibgbgOiqhFPhX2i73hzpc9TAi-/view?usp=sharing)
Baixe esse arquivo compactado e descompacte dentro do arquivo deste projeto.

## Setup do ambiente
Você precisa entrar na pasta deste projeto e execute o seguinte código:
* Python3: `pip3 install -r requirements.txt` (recomendado)
* Python2: `pip install -r requirements.txt`

## Execução do projeto
Após descompactar o arquivo contendo os datasets, cada pasta resultante da descompactação pertence a um dataset utilizado neste experimento. Dentro de cada um terá o arquivo origial (não tratado) e um arquivo chamado data.csv (arquivo já tratado). Além disso, haverá uma pasta chamada **results-ready** contendo os resultados já prontos.

### Análise dos datasets
Caso você queira saber a distribuição das classes de cada dataset basta executar o seguinte comando: 
* Python3: `python3 datasets-stats.py` (recomendado)
* Python2: `python datasets-stats.py`

Obs: Você também pode ver os arquivos **desbalanceamento.xlsx** e **colunas-deletadas-lendingclub.xlsx** para saber mais sobre o desbalanceamento das bases e oscolha das colunas para serem as target além das colunas retiradas da maior base (lending club) respectivamente.

### Tratamento dos dados
Basta apenas executar o seguinte comando:
* Python3: `python3 data-processing.py` (recomendado)
* Python2: `python data-processing.py`
 
Ao executar o comando acima, será gerado um arquivo chamado **data.csv** com o dataset já tratado em cada uma das pastas dos datasets. Como esses arquivos ja existem, eles apenas serão sobescritos.

### Geração dos resultados
Para gerar os resultados é necessário executar o seguinte comando:
* Python3: `python3 main.py` (recomendado)
* Python2: `python main.py`

O resultado da execução acima é a criação de diversos arquivos contendo métricas específicas relativas ao desempenho dos diferentes classificadores em cada uma das bases analisdas. Esses arquivos serão criado no diretório results (atualmente vazio).
