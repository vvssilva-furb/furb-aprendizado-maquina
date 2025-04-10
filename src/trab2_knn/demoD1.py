import scipy.io as scipy
import implementacao_knn as knn

def main():
    mat = scipy.loadmat('grupoDados1.mat')

    grupoTest = mat['grupoTest']
    grupoTrain = mat['grupoTrain']
    testRots = mat['testRots']
    trainRots = mat['trainRots']

    # Previsto: 96%
    rotulo_previsto = knn.meuKnn(grupoTrain, trainRots, grupoTest, 1)
    print(knn.funcao_acuracia(rotulo_previsto, testRots))

    # Previsto: 94%
    rotulo_previsto = knn.meuKnn(grupoTrain, trainRots, grupoTest, 10)
    print(knn.funcao_acuracia(rotulo_previsto, testRots))

    knn.visualizaPontos(grupoTest, testRots, 1, 2)

if __name__ == "__main__":
    main()