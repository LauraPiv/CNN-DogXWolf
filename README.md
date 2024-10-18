# CNN-DogXWolf
## Summary/Sumário
- [1. Description/Descrição](Description)
- [2. Version/Versão](Version)
- [3. Materials/Materiais](materials)
- [4. How to use/Como Usar](how-to-use)
- [6. Copyright and Acknowledgements](copyright)

## 1. Description/Descrição
**[ENG]**
This project is a machine learning model designed to classify images as either wolves or dogs. Using two datasets of 1,000 images each from Kaggle, the model analyzes the images and predicts the likelihood of the animal being a wolf or a dog. It is built with flexibility in mind, allowing it to be extended for a wide range of image classification tasks. While this project specifically focuses on distinguishing between wolves and dogs, the underlying architecture and methods can be adapted for other types of image classification with minimal changes.
The code follows several stages: Reading Data, Training Data, Preparing Training & Testing data for work, Initializing the model along with the input shape, displaying learning curves, making predictions, generating a confusion matrix, calculating Conventional Neural Network Accuracy, Expectation with classes and Image, and a final module where the user can upload a photo, and the code will predict whether it's a dog or a wolf, showing the prediction accuracy.

**[PT-BR]**
Este projeto é um modelo de machine learning desenvolvido para classificar imagens como lobos ou cachorros. Usando dois conjuntos de dados de 1.000 imagens cada, do Kaggle, o modelo analisa as imagens e prevê a probabilidade de o animal ser um lobo ou um cachorro. Ele foi projetado com flexibilidade em mente, permitindo ser estendido para uma ampla gama de tarefas de classificação de imagens. Embora este projeto tenha como foco específico distinguir entre lobos e cachorros, a arquitetura e os métodos subjacentes podem ser adaptados para outros tipos de classificação de imagens com poucas alterações.
O código segue várias etapas: Leitura dos Dados, Treinamento dos Dados, Preparação dos Dados de Treinamento e Teste, Inicialização do modelo juntamente com o formato de entrada, exibição dos gráficos de aprendizagem, realização de previsões, geração da matriz de confusão, cálculo da Acurácia da Rede Neural Convencional, Expectativa com classes e Imagem, e um módulo final onde o usuário pode enviar uma foto, e o código irá prever se é um cachorro ou um lobo, mostrando a acurácia da previsão.

## 2. Version/Versão
**py.3.13.0**

## 3. Materials/Materiais
**[ENG]**
To run this project, you need to have the following tools and libraries installed:
- **Python 3.13.0**
- **Jupyter Notebook** or any Python IDE
- **Libraries**:
  - `numpy` - For numerical operations
  - `pandas` - For data manipulation
  - `matplotlib` - For visualizing data and plots
  - `seaborn` - For enhanced visualizations
  - `Pillow (PIL)` - For image manipulation
  - `tensorflow` or `keras` - For building and training the machine learning model (includes Keras)
  - `scikit-learn` - For model evaluation metrics and data splitting
  - `opencv-python` - For additional image processing capabilities

You can install all the necessary libraries using `pip`:

`pip install numpy pandas matplotlib seaborn pillow tensorflow scikit-learn opencv-python`

**[PT-BR]**
Para executar este projeto, você precisa ter as seguintes ferramentas e bibliotecas instaladas:
- **Python 3.13.0**
- **Jupyter Notebook** ou qualquer IDE Python
- **Bibliotecas:**:
  - `numpy` - Para operações numéricas
  - `pandas` - Para manipulação de dados
  - `matplotlib` - Para visualização de dados e gráficos
  - `seaborn` - Para visualizações aprimoradas
  - `Pillow (PIL)` - Para manipulação de imagens
  - `tensorflow` or `keras` - Para construir e treinar o modelo de machine learning (inclui Keras)
  - `scikit-learn` - Para métricas de avaliação de modelos e divisão de dados
  - `opencv-python` - Para recursos adicionais de processamento de imagens
  
Você pode instalar todas as bibliotecas necessárias usando `pip`:

`pip install numpy pandas matplotlib seaborn pillow tensorflow scikit-learn opencv-python`

## 4. How to use/Como Usar
**[ENG]**
Follow these steps to run the CNN code for classifying wolf and dog images:

1. **Prerequisites**:
   - Ensure you have **Python 3.13.0** installed on your machine.
   - Install **Jupyter Notebook** or any Python IDE (e.g., PyCharm, VSCode).
2. **Clone the Repository**:
   - If your code is hosted on GitHub, clone the repository using:
     ```bash
     git clone <REPOSITORY_URL>
     ```
   - Navigate to the project directory:
     ```bash
     cd project_directory
     ```
3. **Download the Datasets From Kaggle:**
```markdown
[Wolf and Dog Images Dataset] (/kaggle/input/dogs-vs-wolves)
```
Ensure the datasets are placed in the correct directory as specified in your code. Typically, this would be a folder named `data/` or similar within your project directory.

4. **Open the Jupyter Notebook:**
Start Jupyter Notebook with the following command:
```bash
jupyter notebook
```
This will open a web interface in your browser. Navigate to the notebook file `(.ipynb)` that contains your CNN code.

5. **Run the Notebook Cells In Order**
6. **Make Custom Predictions:**
At the end of the notebook, there should be a module that allows you to upload a photo. Use this section to test the model with your own images of wolves or dogs.
The code will display whether the image is classified as a wolf or a dog, along with the prediction accuracy.
7. **Aditional Notes:**
Depending on your code, you may need to adjust the file paths or the hyperparameters of the neural network.
   
**[PT-BR]**
Siga estas etapas para executar o código da CNN para classificar imagens de lobos e cães:

1. **Pré-requisitos**:
- Certifique-se de que você tenha **Python 3.13.0** instalado na sua máquina.
- Instale o **Jupyter Notebook** ou qualquer IDE Python (por exemplo, PyCharm, VSCode).

2. **Clone o Repositório:**
- Se o seu código estiver hospedado no GitHub, clone o repositório usando:
    ```bash
    git clone <REPOSITORY_URL>
    ```
- Navegue até o diretório do projeto:
    ```bash
    cd project_directory
    ```

3. **Baixe os Conjuntos de Dados do Kaggle:**
```markdown
[Conjunto de Dados de Imagens de Lobos e Cães] (/kaggle/input/dogs-vs-wolves)
```

Certifique-se de que os conjuntos de dados estejam colocados no diretório correto, conforme especificado no seu código. Normalmente, isso seria uma pasta chamada `data/` ou semelhante dentro do diretório do seu projeto.

4. **Abra o Jupyter Notebook:**
Inicie o Jupyter Notebook com o seguinte comando:
```bash
jupyter notebook
```
Isso abrirá uma interface web no seu navegador. Navegue até o arquivo do notebook `(.ipynb)` que contém seu código da CNN.

5. **Execute as Células do Notebook em Ordem**
6. **Faça Previsões Personalizadas:**
No final do notebook, deve haver um módulo que permite que você faça upload de uma foto. Use esta seção para testar o modelo com suas próprias imagens de lobos ou cães.
O código exibirá se a imagem é classificada como um lobo ou um cão, juntamente com a precisão da previsão.
7. **Notas Adicionais:**
Dependendo do seu código, você pode precisar ajustar os caminhos dos arquivos ou os hiperparâmetros da rede neural.

## 6. Copyright and Acknowledgements
**[ENG]**
- Copyright

This project and its code are © [LauPiv], [2024]. All rights reserved.

- Acknowledgements

The dataset used in this project was obtained from Kaggle. The original authors are:

[Harish Vutukuri] - [Link to the dataset on Kaggle](</kaggle/input/dogs-vs-wolves>).

Please refer to the dataset's page for more information regarding its usage rights and licensing.

**[PT-BR]**
- Direitos Autorais

Este projeto e seu código são © [LauPiv], [2024]. Todos os direitos reservados.

- Agradecimentos

O conjunto de dados utilizado neste projeto foi obtido do Kaggle. Os autores originais são:

[Harish Vutukuri] - [Link para o conjunto de dados no Kaggle](</kaggle/input/dogs-vs-wolves>).

Por favor, consulte a página do conjunto de dados para mais informações sobre seus direitos de uso e licenciamento.
