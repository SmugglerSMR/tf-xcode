// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Datasets
import ImageClassificationModels
import TensorFlow
import ModelSupport

//let flagMode = "convert"
let flagMode = "predict"

let epochCount = 1
let batchSize = 64
let count = 2

let temporaryDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("MobileNetV2_images")
let path_model = URL(string: "file:///var/folders/x6/qx4_dw3s0jg8jrcr7_0t9sz00000gn/T/MobileNetV2_images/model2.ckpt")
let path_python_model = URL(string: "file:///Users/ramil/Projects/tf-xcode/models/posture_v1_python/cp.ckpt")
let path_image = "/Users/ramil/Projects/posture-xcode/images/good1.jpg"

let imageSize = 192

if flagMode == "train" {
    print("Starting training...")

    let dataset = ImagePosture(
        batchSize: batchSize,
        inputSize: .resized150,
        outputSize: imageSize
    )
    
    //print(dataset)

//    let dataset = Imagenette(
//        batchSize: batchSize,
//        inputSize: .resized320,
//        outputSize: 224
//    )

    var model = PostureNetV1()

    try ImageClassificationSummary(model: model, name: "model2.ckpt")

    let optimizer = SGD(for: model, learningRate: 0.002, momentum: 0.9)

    for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() {
        print("epoch: \(epoch) / \(epochCount)")
        Context.local.learningPhase = .training
        var trainingLossSum: Float = 0
        var trainingBatchCount = 0
        var ii = 0
        for batch in epochBatches {
            print("-> \(ii)")
            let (images, labels) = (batch.data, batch.label)
            let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
                let logits = model(images)
                return softmaxCrossEntropy(logits: logits, labels: labels)
            }
            trainingLossSum += loss.scalarized()
            trainingBatchCount += 1
            optimizer.update(&model, along: gradients)
            ii += 1
        }

        Context.local.learningPhase = .inference
        var testLossSum: Float = 0
        var testBatchCount = 0
        var correctGuessCount = 0
        var totalGuessCount = 0
        for batch in dataset.validation {
            let (images, labels) = (batch.data, batch.label)
            let logits = model(images)
            testLossSum += softmaxCrossEntropy(logits: logits, labels: labels).scalarized()
            testBatchCount += 1

            let correctPredictions = logits.argmax(squeezingAxis: 1) .== labels
            correctGuessCount = correctGuessCount
                + Int(
                    Tensor<Int32>(correctPredictions).sum().scalarized())
            totalGuessCount = totalGuessCount + batch.data.shape[0]
        }

        let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
        print(
            """
            [Epoch \(epoch)] \
            Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy)) \
            Loss: \(testLossSum / Float(testBatchCount))
            """
        )
    }

    try ImageClassificationWriteCheckpoint(model: model, to: temporaryDirectory, name: "model2.ckpt")
    
    // проверим

    guard FileManager.default.fileExists(atPath: path_image) else {
        print("Error: Failed to load image \(path_image). Check that the file exists and is in JPEG format.")
        exit(1)
    }

    let imageTensor = Image(jpeg: URL(fileURLWithPath: path_image)).resized(to: (imageSize, imageSize)).tensor / 255.0

    let out = model(imageTensor.expandingShape(at: 0))

    print(out)
    
    var model1 = PostureNetV1()
    //var model1 = MobileNetV2(classCount: count)
    
    model1.readCheckpoint(to: path_model!, name: "model2.ckpt")

    // Apply the model to image.
    let out1 = model1(imageTensor.expandingShape(at: 0))

    print(out1)


}
else if flagMode == "convert" {
    print("Starting converting...")

    var model = PostureNetV1()
    
    try ImageClassificationSummary(model: model, name: "model2.ckpt")
    
    model.readPythonCheckpoint(to: path_python_model!, name: "model2.ckpt")

    try ImageClassificationWriteCheckpoint(model: model, to: temporaryDirectory, name: "model2.ckpt")
    
    print("Convert to \(temporaryDirectory) successful!")

}
else if flagMode == "predict" {
    print("Starting predicting...")

    guard FileManager.default.fileExists(atPath: path_image) else {
        print("Error: Failed to load image \(path_image). Check that the file exists and is in JPEG format.")
        exit(1)
    }

    let imageTensor = Image(jpeg: URL(fileURLWithPath: path_image)).resized(to: (imageSize, imageSize)).tensor / 255.0

    var model = PostureNetV1()
    
    model.readCheckpoint(to: path_model!, name: "model2.ckpt")

    let out = model(imageTensor.expandingShape(at: 0))

    print(out)

}
else {
    
    //try ImageClassificationReadCheckpoint(model: &model, to: temporaryDirectory, name: "model2.ckpt")
 
    guard FileManager.default.fileExists(atPath: path_image) else {
        print("Error: Failed to load image \(path_image). Check that the file exists and is in JPEG format.")
        exit(1)
    }

    //let imageTensor = Image(jpeg: URL(fileURLWithPath: path_image)).resized(to: (imageSize, imageSize)).tensor / 255.0
    
//    let resizedInput = resize(
//        images: input, size: (newHeight, newWidth), method: .nearest)
//    return resizedInput.sequenced(through: reflectionPad, conv2d)
    
    
//    let map = [
//        "conv1":  ("0", [3, 3, 3, 16]),
//        "conv2":  ("1", [3, 3, 16, 32]),
//        "conv3":  ("2", [3, 3, 32, 64]),
//        "dense1": ("3", [20736, 512]),
//        "dense2": ("4", [512, 2]),
//
//    ]

    //try ImageClassificationSummary(model: model, name: "model2.ckpt")

    //try ImageClassificationReadCheckpoint(model: model, to: path_model!, name: "model2.ckpt")

    //try ImageClassificationPythonRead(model: model, to: path_python_model!, name: "model2.ckpt")
    
    //model.readCheckpoint(to: path_model!, name: "model2.ckpt")
    //model.readPythonCheckpoint(to: path_python_model!, name: "model2.ckpt")
     

    // Apply the model to image.
    //let out = model(imageTensor.expandingShape(at: 0))

    //print(out)
 
    let imageTensor = Image(jpeg: URL(fileURLWithPath: path_image)).resized(to: (imageSize, imageSize)).tensor / 255.0
    //print(imageTensor)
    
    var model1 = PostureNetV1()
    //var model1 = MobileNetV2(classCount: count)
    
    try ImageClassificationSummary(model: model1, name: "model2.ckpt")
    
    //model1.readCheckpoint(to: path_model!, name: "model2.ckpt")
    model1.readPythonCheckpoint(to: path_python_model!, name: "model2.ckpt")

    // Apply the model to image.
    let out1 = model1(imageTensor.expandingShape(at: 0))

    print(out1[0])

    
}


//let recreatedmodel = try GPT2(checkpoint: temporaryDirectory.appendingPathComponent("model2.ckpt"))
