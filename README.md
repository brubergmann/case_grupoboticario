# Case Boticário - Previsão de Vendas
## Desenvolvedora: Bruna Moura Bergmann

## Features

- Treina e salva o modelo com a base histórica de vendas
- Realiza a previsão utilizando o modelo treinado e salva no arquivo resultado.xlsx

## Pré-requisitos

- Python versão 3.9 ou superior instalado
- Módulos do Python instalados: numpy, pandas, scikit-learn, seaborn, sqlite3, openpyxl

## Utilização

1. Copiar os arquivos recebidos para um diretório (<INST_DIR>)
2. Abrir o prompt de comando e executar as seguintes instruções

```sh
cd <INST_DIR>
python caseBoticario.py
```


3. Os resultados serão gravados nos arquivos
<INST_DIR>/resultado.xlsx  : Resultado da predição
<INST_DIR>/modelo_dump.pkl  : Modelo treinado


## Formato dos dados

> `COD_MATERIAL`           Código do produto
> `COD_CICLO`               Ciclo (variável de tempo)
> `DES_CATEGORIA_MATERIAL`   Categoria do produto   
> `DES_MARCA_MATERIAL`  Marca do Produto
> `FLG_DATA`    Indicador de existência ou não de datas comemorativas no ciclo
> `FLG_CAMPANHA_MKT_A` Indicador de existência ou não da campanha de marketing A no ciclo
> `FLG_CAMPANHA_MKT_B`  Indicador de existência ou não da campanha de marketing B no ciclo          
> `FLG_CAMPANHA_MKT_C` Indicador de existência ou não da campanha de marketing C no ciclo
> `FLG_CAMPANHA_MKT_D` Indicador de existência ou não da campanha de marketing D no ciclo
> `FLG_CAMPANHA_MKT_E` Indicador de existência ou não da campanha de marketing E no ciclo
> `FLG_CAMPANHA_MKT_F` Indicador de existência ou não da campanha de marketing F no ciclo
> `PCT_DESCONTO` Percentual de desconto aplicado na venda (0 - 100)
> `VL_PRECO`  Preço do produto
> `QT_VENDA`  Quantidade de Vendas do produto (vazio para os itens a prever)

