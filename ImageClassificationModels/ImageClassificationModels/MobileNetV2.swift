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

import TensorFlow
import ModelSupport

// Original Paper:
// "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
// Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
// https://arxiv.org/abs/1801.04381

fileprivate func makeDivisible(filter: Int, widthMultiplier: Float = 1.0, divisor: Float = 8.0)
    -> Int
{
    /// Return a filter multiplied by width, evenly divisible by the divisor
    let filterMult = Float(filter) * widthMultiplier
    let filterAdd = Float(filterMult) + (divisor / 2.0)
    var div = filterAdd / divisor
    div.round(.down)
    div = div * Float(divisor)
    var newFilterCount = max(1, Int(div))
    if newFilterCount < Int(0.9 * Float(filter)) {
        newFilterCount += Int(divisor)
    }
    return Int(newFilterCount)
}

fileprivate func roundFilterPair(filters: (Int, Int), widthMultiplier: Float) -> (Int, Int) {
    return (
        makeDivisible(filter: filters.0, widthMultiplier: widthMultiplier),
        makeDivisible(filter: filters.1, widthMultiplier: widthMultiplier)
    )
}






public struct MobileNetV2: Layer {
    @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
    public var inputConv: Conv2D<Float>
    public var inputConvBatchNorm: BatchNorm<Float>
    public var initialInvertedBottleneck: InitialInvertedBottleneckBlock

    public var residualBlockStack1: InvertedBottleneckBlockStack
    public var residualBlockStack2: InvertedBottleneckBlockStack
    public var residualBlockStack3: InvertedBottleneckBlockStack
    public var residualBlockStack4: InvertedBottleneckBlockStack
    public var residualBlockStack5: InvertedBottleneckBlockStack

    public var invertedBottleneckBlock16: InvertedBottleneckBlock

    public var outputConv: Conv2D<Float>
    public var outputConvBatchNorm: BatchNorm<Float>
    public var avgPool = GlobalAvgPool2D<Float>()
    public var outputClassifier: Dense<Float>
                                

    public init(classCount: Int = 1000, widthMultiplier: Float = 1.0) {
        inputConv = Conv2D<Float>(
            filterShape: (3, 3, 3, makeDivisible(filter: 32, widthMultiplier: widthMultiplier)),
            strides: (2, 2),
            padding: .valid)
        inputConvBatchNorm = BatchNorm(
            featureCount: makeDivisible(filter: 32, widthMultiplier: widthMultiplier))

        initialInvertedBottleneck = InitialInvertedBottleneckBlock(
            filters: (32, 16), widthMultiplier: widthMultiplier)

        residualBlockStack1 = InvertedBottleneckBlockStack(
            filters: (16, 24), widthMultiplier: widthMultiplier, blockCount: 2)
        residualBlockStack2 = InvertedBottleneckBlockStack(
            filters: (24, 32), widthMultiplier: widthMultiplier, blockCount: 3)
        residualBlockStack3 = InvertedBottleneckBlockStack(
            filters: (32, 64), widthMultiplier: widthMultiplier, blockCount: 4)
        residualBlockStack4 = InvertedBottleneckBlockStack(
            filters: (64, 96), widthMultiplier: widthMultiplier, blockCount: 3,
            initialStrides: (1, 1))
        residualBlockStack5 = InvertedBottleneckBlockStack(
            filters: (96, 160), widthMultiplier: widthMultiplier, blockCount: 3)

        invertedBottleneckBlock16 = InvertedBottleneckBlock(
            filters: (160, 320), widthMultiplier: widthMultiplier)

        var lastBlockFilterCount = makeDivisible(filter: 1280, widthMultiplier: widthMultiplier)
        if widthMultiplier < 1 {
            // paper: "One minor implementation difference, with [arxiv:1704.04861] is that for
            // multipliers less than one, we apply width multiplier to all layers except the very
            // last convolutional layer."
            lastBlockFilterCount = 1280
        }

        outputConv = Conv2D<Float>(
            filterShape: (
                1, 1,
                makeDivisible(filter: 320, widthMultiplier: widthMultiplier), lastBlockFilterCount
            ),
            strides: (1, 1),
            padding: .same)
        outputConvBatchNorm = BatchNorm(featureCount: lastBlockFilterCount)

        outputClassifier = Dense(
            inputSize: lastBlockFilterCount, outputSize: classCount)
    }

    public mutating func readCheckpoint(to location: URL, name: String) {
        print("MobileNetV2:readCheckpoint: \(name)  URL: \(location.path)")
        
        // Try loading from the given checkpoint.
        do {
            let auxiliary: [String] = [
                "checkpoint",
                "encoder.json",
                "hparams.json",
                "model.ckpt.meta",
                "vocab.bpe",
            ]

            let reader: CheckpointReader = try CheckpointReader(
                checkpointLocation: location,
                modelName: name,
                additionalFiles: auxiliary)
            // TODO(michellecasbon): expose this.
            reader.isCRCVerificationEnabled = false
            
            //print(reader)
            
            //let storage: URL = reader.localCheckpointLocation.deletingLastPathComponent()
            
            let parameters = TransformerLMConfig(
                        vocabSize: 1,
                        contextSize: 1024,
                        embeddingSize: 768,
                        headCount: 12,
                        layerCount: 12)

            
            //print(storage)
            
            load_weight(reader: reader, config: parameters, scope: "model")
 
            
            print("MobileNetV2 loaded from checkpoint successfully.")
        } catch {
            // If checkpoint is invalid, throw the error and exit.
            print("Fail to load MobileNetV2 from checkpoint. \(error)")
        }

    }

        
    public mutating func readPythonCheckpoint(to location: URL, name: String) {
        print("MobileNetV2:readPythonCheckpoint: \(name)  URL: \(location.path)")
        
        // Try loading from the given checkpoint.
        do {
            let auxiliary: [String] = [
                "checkpoint",
                "encoder.json",
                "hparams.json",
                "model.ckpt.meta",
                "vocab.bpe",
            ]

            let reader: CheckpointReader = try CheckpointReader(
                checkpointLocation: location,
                modelName: name,
                additionalFiles: auxiliary)
            // TODO(michellecasbon): expose this.
            reader.isCRCVerificationEnabled = false
            
            //reader.printMetadata()
            
            //let storage: URL = reader.localCheckpointLocation.deletingLastPathComponent()
            
            //print(storage)
            
            var numberLayers = 0
            
            load_python_weight(reader: reader, scope: "model", number: &numberLayers)
             
            print("MobileNetV2 loaded from python checkpoint successfully. \(numberLayers)")
        } catch {
            // If checkpoint is invalid, throw the error and exit.
            print("Fail to load MobileNetV2 from python checkpoint. \(error)")
        }

    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved = relu6(input.sequenced(through: zeroPad, inputConv, inputConvBatchNorm))
        let initialConv = initialInvertedBottleneck(convolved)
        let backbone = initialConv.sequenced(
            through: residualBlockStack1, residualBlockStack2, residualBlockStack3,
            residualBlockStack4, residualBlockStack5)
        let output = relu6(outputConvBatchNorm(outputConv(invertedBottleneckBlock16(backbone))))
        return output.sequenced(through: avgPool, outputClassifier)
    }
}

extension MobileNetV2: ExportableLayer {
    var nameMappings: [String: String] {
        [
         //"zeroPad": "zeroPad",
         "inputConv": "inputConv",
         "inputConvBatchNorm": "inputConvBN",
         "initialInvertedBottleneck": "initBottleneck",
         "residualBlockStack1": "residual1",
         "residualBlockStack2": "residual2",
         "residualBlockStack3": "residual3",
         "residualBlockStack4": "residual4",
         "residualBlockStack5": "residual5",
         "invertedBottleneckBlock16": "Bottleneck16",
         "outputConv": "outputConv",
         "outputConvBatchNorm": "outputConvBN",
         "avgPool": "avgPool",
         "outputClassifier": "outputClassifier",
        ]
    }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        print("--MobileNetV2:save_weight-- \(scope)")

        inputConv.save_weight(scope: scope+"/inputConv", tensors: &tensors)
        inputConvBatchNorm.save_weight(scope: scope+"/inputConvBN", tensors: &tensors)
        initialInvertedBottleneck.save_weight(scope: scope+"/initBottleneck", tensors: &tensors)

        residualBlockStack1.save_weight(scope: scope+"/residual1", tensors: &tensors)
        residualBlockStack2.save_weight(scope: scope+"/residual2", tensors: &tensors)
        residualBlockStack3.save_weight(scope: scope+"/residual3", tensors: &tensors)
        residualBlockStack4.save_weight(scope: scope+"/residual4", tensors: &tensors)
        residualBlockStack5.save_weight(scope: scope+"/residual5", tensors: &tensors)
        
        invertedBottleneckBlock16.save_weight(scope: scope+"/Bottleneck16", tensors: &tensors)

        outputConv.save_weight(scope: scope+"/outputConv", tensors: &tensors)
        outputConvBatchNorm.save_weight(scope: scope+"/outputConvBN", tensors: &tensors)
        //avgPool.save_weight(scope: scope+"/avgPool", tensors: &tensors)
        outputClassifier.save_weight(scope: scope+"/outputClassifier", tensors: &tensors)

    }
}

extension MobileNetV2: ImportableLayer {
    mutating public func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        print("--MobileNetV2:load_weight-- \(scope)")
        
        inputConv.load_weight(reader: reader, config: config, scope: scope+"/inputConv")
        inputConvBatchNorm.load_weight(reader: reader, config: config, scope: scope+"/inputConvBN")
        initialInvertedBottleneck.load_weight(reader: reader, config: config, scope: scope+"/initBottleneck")

        residualBlockStack1.load_weight(reader: reader, config: config, scope: scope+"/residual1")
        residualBlockStack2.load_weight(reader: reader, config: config, scope: scope+"/residual2")
        residualBlockStack3.load_weight(reader: reader, config: config, scope: scope+"/residual3")
        residualBlockStack4.load_weight(reader: reader, config: config, scope: scope+"/residual4")
        residualBlockStack5.load_weight(reader: reader, config: config, scope: scope+"/residual5")
        
        invertedBottleneckBlock16.load_weight(reader: reader, config: config, scope: scope+"/Bottleneck16")

        outputConv.load_weight(reader: reader, config: config, scope: scope+"/outputConv")
        outputConvBatchNorm.load_weight(reader: reader, config: config, scope: scope+"/outputConvBN")
        outputClassifier.load_weight(reader: reader, config: config, scope: scope+"/outputClassifier")

    }
}


extension MobileNetV2: SummaryLayer {
    public func summary(scope: String) {
        //print("--MobileNetV2:summary-- \(scope)")
        
        print("MobileNetV2:                                                            \(scope)")
        print("-------------")

        zeroPad.summary(scope: scope+"/inputPad")
        inputConv.summary(scope: scope+"/inputConv")
        inputConvBatchNorm.summary(scope: scope+"/inputConvBN")
        initialInvertedBottleneck.summary(scope: scope+"/initBottleneck")

        residualBlockStack1.summary(scope: scope+"/residual1")
        residualBlockStack2.summary(scope: scope+"/residual2")
        residualBlockStack3.summary(scope: scope+"/residual3")
        residualBlockStack4.summary(scope: scope+"/residual4")
        residualBlockStack5.summary(scope: scope+"/residual5")
        
        invertedBottleneckBlock16.summary(scope: scope+"/Bottleneck16")

        outputConv.summary(scope: scope+"/outputConv")
        outputConvBatchNorm.summary(scope: scope+"/outputConvBN")
        avgPool.summary(scope: scope+"/avgPool")
        outputClassifier.summary(scope: scope+"/outputClassifier")

    }
}

extension MobileNetV2: ImportablePythonLayer {
    mutating public func load_python_weight(reader: CheckpointReader, scope: String, number: inout Int) {
        print("--MobileNetV2:load_python_weight-- \(scope)")
        
        inputConv.load_python_weight(reader: reader, scope: scope+"/inputConv", number: &number)
        inputConvBatchNorm.load_python_weight(reader: reader, scope: scope+"/inputConvBN", number: &number)
        initialInvertedBottleneck.load_python_weight(reader: reader, scope: scope+"/initBottleneck", number: &number)

        residualBlockStack1.load_python_weight(reader: reader, scope: scope+"/residual1", number: &number)
        residualBlockStack2.load_python_weight(reader: reader, scope: scope+"/residual2", number: &number)
        residualBlockStack3.load_python_weight(reader: reader, scope: scope+"/residual3", number: &number)
        residualBlockStack4.load_python_weight(reader: reader, scope: scope+"/residual4", number: &number)
        residualBlockStack5.load_python_weight(reader: reader, scope: scope+"/residual5", number: &number)
        
        invertedBottleneckBlock16.load_python_weight(reader: reader, scope: scope+"/Bottleneck16", number: &number)

        outputConv.load_python_weight(reader: reader, scope: scope+"/outputConv", number: &number)
        outputConvBatchNorm.load_python_weight(reader: reader, scope: scope+"/outputConvBN", number: &number)
        outputClassifier.load_python_weight(reader: reader, scope: scope+"/outputClassifier", number: &number)

    }
}
