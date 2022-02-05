//
//  main.swift
//  walkthrough
//
//  Created by Рамиль Садыков on 09.06.2020.
//  Copyright © 2020 Sadukow. All rights reserved.
//
//  https://www.tensorflow.org/swift/tutorials/model_training_walkthrough?hl=IT
//  https://github.com/tensorflow/swift/blob/master/docs/site/tutorials/model_training_walkthrough.ipynb
//
//  Документация: https://github.com/tensorflow/swift/tree/master/docs

import Foundation
//import FoundationNetworking
//import ModelSupport

import TensorFlow
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

print("Hello, TensorFlow!")

let plt = Python.import("matplotlib.pyplot")

// Download a helper file that helps us work around some temporary limitations
// in the dataset API.
func download(from sourceString: String, to destinationString: String) {
    let source = URL(string: sourceString)!
    let destination = URL(fileURLWithPath: destinationString)
    let data = try! Data.init(contentsOf: source)
    try! data.write(to: destination)
    print("download successfull \(destination)!")
}
/*download(
    from: "https://raw.githubusercontent.com/tensorflow/swift/master/docs/site/tutorials/TutorialDatasetCSVAPI.swift",
    to: "TutorialDatasetCSVAPI.swift")
*/
// Download the dataset

let trainDataFilename = "iris_training.csv"
//download(from: "http://download.tensorflow.org/data/iris_training.csv", to: trainDataFilename)

// Inspect the data
let f = Python.open(trainDataFilename)
for _ in 0..<5 {
    print(Python.next(f).strip())
}
f.close()

let featureNames = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
let labelName = "species"
let columnNames = featureNames + [labelName]

print("Features: \(featureNames)")
print("Label: \(labelName)")

let classNames = ["Iris setosa", "Iris versicolor", "Iris virginica"]

// Create a Dataset

let batchSize = 32

/// A batch of examples from the iris dataset.
struct IrisBatch {
    /// [batchSize, featureCount] tensor of features.
    let features: Tensor<Float>

    /// [batchSize] tensor of labels.
    let labels: Tensor<Int32>
}


let trainDataset: Dataset<IrisBatch> = Dataset(
    contentsOfCSVFile: trainDataFilename, hasHeader: true,
    featureColumns: [0, 1, 2, 3], labelColumns: [4]
).batched(batchSize)


let firstTrainExamples = trainDataset.first!
let firstTrainFeatures = firstTrainExamples.features
let firstTrainLabels = firstTrainExamples.labels
print("First batch of features: \(firstTrainFeatures)")
print("First batch of labels: \(firstTrainLabels)")

let firstTrainFeaturesTransposed = firstTrainFeatures.transposed()
let petalLengths = firstTrainFeaturesTransposed[2].scalars
let sepalLengths = firstTrainFeaturesTransposed[0].scalars

/*
plt.scatter(petalLengths, sepalLengths, c: firstTrainLabels.array.scalars)
plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()
*/

//Создайте модель с помощью библиотеки глубокого обучения Swift for TensorFlow

//Библиотека глубокого обучения Swift for TensorFlow определяет примитивные слои и соглашения для их соединения, что облегчает создание моделей и экспериментов.

//Модель - это структура, которая соответствует Layer, что означает, что она определяет метод callAsFunction (_ :), который отображает входные Tensors на выходные Tensors. Метод callAsFunction (_ :) часто просто упорядочивает ввод через подслои. Давайте определим IrisModel, которая последовательно вводит данные через три плотных подслоя.

let hiddenSize: Int = 10
struct IrisModel: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: 3)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}

var model = IrisModel()

//let remoteCheckpoint = URL(string: "https://storage.googleapis.com/s4tf-hosted-binaries/checkpoints/VGG16/vgg_16.tar.gz")!
//let checkpointReader = try CheckpointReader(checkpointLocation: remoteCheckpoint, modelName: "VGG")

// Train the model

let untrainedLogits = model(firstTrainFeatures)
let untrainedLoss = softmaxCrossEntropy(logits: untrainedLogits, labels: firstTrainLabels)
print("Loss test: \(untrainedLoss)")



let optimizer = SGD(for: model, learningRate: 0.01)

let (loss, grads) = valueWithGradient(at: model) { model -> Tensor<Float> in
    let logits = model(firstTrainFeatures)
    return softmaxCrossEntropy(logits: logits, labels: firstTrainLabels)
}
print("Current loss: \(loss)")

// Training loop

let epochCount = 50
var trainAccuracyResults: [Float] = []
var trainLossResults: [Float] = []

func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
    return Tensor<Float>(predictions .== truths).mean().scalarized()
}

for epoch in 1...epochCount {
    print("epoch: \(epoch)")
    var epochLoss: Float = 0
    var epochAccuracy: Float = 0
    var batchCount: Int = 0
    for batch in trainDataset {
        let (loss, grad) = valueWithGradient(at: model) { (model: IrisModel) -> Tensor<Float> in
            let logits = model(batch.features)
            return softmaxCrossEntropy(logits: logits, labels: batch.labels)
        }
        optimizer.update(&model, along: grad)
        
        let logits = model(batch.features)
        epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: batch.labels)
        epochLoss += loss.scalarized()
        batchCount += 1
    }
    epochAccuracy /= Float(batchCount)
    epochLoss /= Float(batchCount)
    trainAccuracyResults.append(epochAccuracy)
    trainLossResults.append(epochLoss)
    if epoch % 50 == 0 {
        print("Epoch \(epoch): Loss: \(epochLoss), Accuracy: \(epochAccuracy)")
    }
}

// Visualize the loss function over time
/*
plt.figure(figsize: [12, 8])

let accuracyAxes = plt.subplot(2, 1, 1)
accuracyAxes.set_ylabel("Accuracy")
accuracyAxes.plot(trainAccuracyResults)

let lossAxes = plt.subplot(2, 1, 2)
lossAxes.set_ylabel("Loss")
lossAxes.set_xlabel("Epoch")
lossAxes.plot(trainLossResults)

plt.show()
*/


/*

// Use the trained model to make predictions

let unlabeledDataset: Tensor<Float> =
    [[5.1, 3.3, 1.7, 0.5],
     [5.9, 3.0, 4.2, 1.5],
     [6.9, 3.1, 5.4, 2.1]]

let unlabeledDatasetPredictions = model(unlabeledDataset)

for i in 0..<unlabeledDatasetPredictions.shape[0] {
    let logits = unlabeledDatasetPredictions[i]
    let classIdx = logits.argmax().scalar!
    print("Example \(i) prediction: \(classNames[Int(classIdx)]) (\(softmax(logits)))")
}
*/
