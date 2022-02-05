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
//  первый вариант модели для определения осанки:
//  основа - "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
//    +    - Dense(2)
//

fileprivate func makeDivisible(filter: Int, alpha: Float = 1.0, divisor: Float = 8.0)
    -> Int
{
    /// Return a filter multiplied by width, evenly divisible by the divisor
    let filterMult = Float(filter) * alpha
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

fileprivate func roundFilterPair(filters: (Int, Int), alpha: Float) -> (Int, Int) {
    return (
        makeDivisible(filter: filters.0, alpha: alpha),
        makeDivisible(filter: filters.1, alpha: alpha)
    )
}

public struct PostureNetV2: Layer {
    @noDerivative
    public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
    public var inputConv: Conv2D<Float>
    public var inputConvBatchNorm: BatchNorm<Float>

    public var invertedBlock0: Inverted_ResNet_Block
    public var invertedBlock1: Inverted_ResNet_Block
    public var invertedBlock2: Inverted_ResNet_Block
    public var invertedBlock3: Inverted_ResNet_Block
    public var invertedBlock4: Inverted_ResNet_Block
    public var invertedBlock5: Inverted_ResNet_Block
    public var invertedBlock6: Inverted_ResNet_Block
    public var invertedBlock7: Inverted_ResNet_Block
    public var invertedBlock8: Inverted_ResNet_Block
    public var invertedBlock9: Inverted_ResNet_Block
    public var invertedBlock10: Inverted_ResNet_Block
    public var invertedBlock11: Inverted_ResNet_Block
    public var invertedBlock12: Inverted_ResNet_Block
    public var invertedBlock13: Inverted_ResNet_Block
    public var invertedBlock14: Inverted_ResNet_Block
    public var invertedBlock15: Inverted_ResNet_Block
    public var invertedBlock16: Inverted_ResNet_Block

    public var outputConv: Conv2D<Float>
    public var outputConvBatchNorm: BatchNorm<Float>
    public var avgPool = GlobalAvgPool2D<Float>()
    public var dropoutLayer: Dropout<Float>
    public var outputClassifier: Dense<Float>
    
    //public var flatten = Flatten<Float>()
    //public var outputAvgPool = GlobalAvgPool2D<Float>()
    //public var output: Dense<Float>

    
    public init(classCount: Int = 2, alpha: Float = 1.0) {
        
        print("PostureNetV2:init   classCount: \(classCount)    alpha: \(alpha)")
        
        inputConv = Conv2D<Float>(
            filterShape: (3, 3, 3, makeDivisible(filter: 32, alpha: alpha)),
            strides: (2, 2),
            padding: .valid)

        inputConvBatchNorm = BatchNorm(featureCount: makeDivisible(filter: 32, alpha: alpha),
                                       axis: -1,
                                       momentum: 0.999,
                                       epsilon: 0.001)

        invertedBlock0 = Inverted_ResNet_Block( filters: (32, 16),
                                                alpha: alpha,
                                                expansion: 1,
                                                strides: (1, 1),
                                                id: 0 )

        invertedBlock1 = Inverted_ResNet_Block( filters: (16, 24),
                                                alpha: alpha,
                                                expansion: 6,
                                                strides: (2, 2),
                                                id: 1 )
        invertedBlock2 = Inverted_ResNet_Block( filters: (24, 24),
                                                alpha: alpha,
                                                expansion: 6,
                                                strides: (1, 1),
                                                id: 2 )

        invertedBlock3 = Inverted_ResNet_Block( filters: (24, 32),
                                                alpha: alpha,
                                                expansion: 6,
                                                strides: (2, 2),
                                                id: 3 )
        invertedBlock4 = Inverted_ResNet_Block( filters: (32, 32),
                                                alpha: alpha,
                                                expansion: 6,
                                                strides: (1, 1),
                                                id: 4 )
        invertedBlock5 = Inverted_ResNet_Block( filters: (32, 32),
                                                alpha: alpha,
                                                expansion: 6,
                                                strides: (1, 1),
                                                id: 5 )

        invertedBlock6 = Inverted_ResNet_Block( filters: (32, 64),
                                                alpha: alpha,
                                                expansion: 6,
                                                strides: (2, 2),
                                                id: 6 )
        invertedBlock7 = Inverted_ResNet_Block( filters: (64, 64),
                                                alpha: alpha,
                                                expansion: 6,
                                                strides: (1, 1),
                                                id: 7 )
        invertedBlock8 = Inverted_ResNet_Block( filters: (64, 64),
                                                alpha: alpha,
                                                expansion: 6,
                                                strides: (1, 1),
                                                id: 8 )
        invertedBlock9 = Inverted_ResNet_Block( filters: (64, 64),
                                                alpha: alpha,
                                                expansion: 6,
                                                strides: (1, 1),
                                                id: 9 )

        invertedBlock10 = Inverted_ResNet_Block( filters: (64, 96),
                                                 alpha: alpha,
                                                 expansion: 6,
                                                 strides: (1, 1),
                                                 id: 10 )
        invertedBlock11 = Inverted_ResNet_Block( filters: (96, 96),
                                                 alpha: alpha,
                                                 expansion: 6,
                                                 strides: (1, 1),
                                                 id: 11 )
        invertedBlock12 = Inverted_ResNet_Block( filters: (96, 96),
                                                 alpha: alpha,
                                                 expansion: 6,
                                                 strides: (1, 1),
                                                 id: 12 )

        invertedBlock13 = Inverted_ResNet_Block( filters: (96, 160),
                                                 alpha: alpha,
                                                 expansion: 6,
                                                 strides: (2, 2),
                                                 id: 13 )
        
        invertedBlock14 = Inverted_ResNet_Block( filters: (160, 160),
                                                 alpha: alpha,
                                                 expansion: 6,
                                                 strides: (1, 1),
                                                 id: 14 )
        
        invertedBlock15 = Inverted_ResNet_Block( filters: (160, 160),
                                                 alpha: alpha,
                                                 expansion: 6,
                                                 strides: (1, 1),
                                                 id: 15 )
        
        invertedBlock16 = Inverted_ResNet_Block( filters: (160, 320),
                                                 alpha: alpha,
                                                 expansion: 6,
                                                 strides: (1, 1),
                                                 id: 16 )

        var lastBlockFilterCount = makeDivisible(filter: 1280, alpha: alpha)
        if alpha < 1 {
            // paper: "One minor implementation difference, with [arxiv:1704.04861] is that for
            // multipliers less than one, we apply width multiplier to all layers except the very
            // last convolutional layer."
            lastBlockFilterCount = 1280
        }

        outputConv = Conv2D<Float>( filterShape: (1, 1, makeDivisible(filter: 320, alpha: alpha), lastBlockFilterCount),
                                    strides: (1, 1),
                                    padding: .same)
        outputConvBatchNorm = BatchNorm(featureCount: lastBlockFilterCount,
                                        axis: -1,
                                        momentum: 0.999,
                                        epsilon: 0.001)
        
        dropoutLayer = Dropout<Float>(probability: 0.2)

        outputClassifier = Dense(inputSize: lastBlockFilterCount, outputSize: classCount)
    }
    


    public mutating func readCheckpoint(to location: URL, name: String) {
        print("PostureNetV2:readCheckpoint: \(name)  URL: \(location.path)")
        
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
                        
            let parameters = TransformerLMConfig(
                        vocabSize: 1,
                        contextSize: 1024,
                        embeddingSize: 768,
                        headCount: 12,
                        layerCount: 12)

            
            //reader.printMetadata()
            
            load_weight(reader: reader, config: parameters, scope: "model")
 
            print("PostureNetV2 loaded from checkpoint successfully.")
        } catch {
            // If checkpoint is invalid, throw the error and exit.
            print("Fail to load PostureNetV2 from checkpoint. \(error)")
        }

    }

    
    public mutating func readPythonCheckpoint(to location: URL, name: String) {
        print("\nPostureNetV2:readPythonCheckpoint: \nURL: \(location.path)")
        
        // Try loading from the given checkpoint.
        do {
            let auxiliary: [String] = [ "checkpoint" ]

            let reader: CheckpointReader = try CheckpointReader(
                checkpointLocation: location,
                modelName: name,
                additionalFiles: auxiliary)
            reader.isCRCVerificationEnabled = false
            
            //reader.printMetadata()
            
            var numberLayers = 0
            
            load_python_weight(reader: reader, scope: "", number: &numberLayers)
             
            print("PostureNetV2 loaded from python checkpoint  \(numberLayers) layers!\n")
        } catch {
            // If checkpoint is invalid, throw the error and exit.
            print("Fail to load PostureNetV2 from python checkpoint. \(error)")
        }

    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved = relu6(input.sequenced(through: zeroPad, inputConv, inputConvBatchNorm))
        let initialConv = invertedBlock0(convolved)
        let backbone1 = initialConv.sequenced(through: invertedBlock1, invertedBlock2, invertedBlock3, invertedBlock4)
        let backbone2 = backbone1.sequenced(through: invertedBlock5, invertedBlock6, invertedBlock7, invertedBlock8)
        let backbone3 = backbone2.sequenced(through: invertedBlock9, invertedBlock10, invertedBlock11, invertedBlock12)
        let backbone4 = backbone3.sequenced(through: invertedBlock13, invertedBlock14, invertedBlock15, invertedBlock16)
        
        let out1 = relu6(outputConvBatchNorm(outputConv(backbone4)))
        
        return out1.sequenced(through: avgPool, dropoutLayer, outputClassifier)
    }
}


extension PostureNetV2: SummaryLayer {
    public func summary(scope: String) {
        //print("--PostureNetV2:summary-- \(scope)")
        
        print("PostureNetV2:                                                          \(scope)")
        print("-------------")

        zeroPad.summary(scope: scope+"/inputPad")
        inputConv.summary(scope: scope+"/inputConv")
        inputConvBatchNorm.summary(scope: scope+"/inputConvBN")
        ReLU_summary(scope: scope+"/inputReLU")

        invertedBlock0.summary(scope: scope+"/")
        invertedBlock1.summary(scope: scope+"/")
        invertedBlock2.summary(scope: scope+"/")
        invertedBlock3.summary(scope: scope+"/")
        invertedBlock4.summary(scope: scope+"/")
        invertedBlock5.summary(scope: scope+"/")
        invertedBlock6.summary(scope: scope+"/")
        invertedBlock7.summary(scope: scope+"/")
        invertedBlock8.summary(scope: scope+"/")
        invertedBlock9.summary(scope: scope+"/")
        invertedBlock10.summary(scope: scope+"/")
        invertedBlock11.summary(scope: scope+"/")
        invertedBlock12.summary(scope: scope+"/")
        invertedBlock13.summary(scope: scope+"/")
        invertedBlock14.summary(scope: scope+"/")
        invertedBlock15.summary(scope: scope+"/")
        invertedBlock16.summary(scope: scope+"/")

        outputConv.summary(scope: scope+"/outputConv")
        outputConvBatchNorm.summary(scope: scope+"/outputConvBN")
        ReLU_summary(scope: scope+"/outputConvReLU")

        avgPool.summary(scope: scope+"/avgPool")
        outputClassifier.summary(scope: scope+"/outputClassifier")

        //outputAvgPool.summary(scope: scope+"/outputAvgPool")
        //output.summary(scope: scope+"/output")

    }
}

extension PostureNetV2: ExportableLayer {
    var nameMappings: [String: String] {
        [
         "inputConv": "inputConv",
         "inputConvBatchNorm": "inputConvBN",
         "outputConv": "outputConv",
         "outputConvBatchNorm": "outputConvBN",
         "avgPool": "avgPool",
         "outputClassifier": "outputClassifier",
        ]
    }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        print("--PostureNetV2:save_weight-- \(scope)")

        inputConv.save_weight(scope: scope+"/inputConv", tensors: &tensors)
        inputConvBatchNorm.save_weight(scope: scope+"/inputConvBN", tensors: &tensors)
        
        invertedBlock0.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock1.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock2.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock3.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock4.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock5.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock6.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock7.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock8.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock9.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock10.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock11.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock12.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock13.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock14.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock15.save_weight(scope: scope+"/", tensors: &tensors)
        invertedBlock16.save_weight(scope: scope+"/", tensors: &tensors)
        
        outputConv.save_weight(scope: scope+"/outputConv", tensors: &tensors)
        outputConvBatchNorm.save_weight(scope: scope+"/outputConvBN", tensors: &tensors)
        outputClassifier.save_weight(scope: scope+"/outputClassifier", tensors: &tensors)

        //output.save_weight(scope: scope+"/output", tensors: &tensors)

    }
}

extension PostureNetV2: ImportableLayer {
    mutating public func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        print("--PostureNetV2:load_weight-- \(scope)")
        
        inputConv.load_weight(reader: reader, config: config, scope: scope+"/inputConv")
        inputConvBatchNorm.load_weight(reader: reader, config: config, scope: scope+"/inputConvBN")

        invertedBlock0.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock1.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock2.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock3.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock4.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock5.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock6.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock7.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock8.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock9.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock10.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock11.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock12.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock13.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock14.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock15.load_weight(reader: reader, config: config, scope: scope+"/")
        invertedBlock16.load_weight(reader: reader, config: config, scope: scope+"/")

        outputConv.load_weight(reader: reader, config: config, scope: scope+"/outputConv")
        outputConvBatchNorm.load_weight(reader: reader, config: config, scope: scope+"/outputConvBN")
        outputClassifier.load_weight(reader: reader, config: config, scope: scope+"/outputClassifier")

        //output.load_weight(reader: reader, config: config, scope: scope+"/output")
    }
}

extension PostureNetV2: ImportablePythonLayer {
    mutating public func load_python_weight(reader: CheckpointReader, scope: String, number: inout Int) {
        //print("--PostureNetV2:load_python_weight-- \(scope)")
        
        number = 0
        var n = 0
        var prfx = "layer_with_weights-\(n)/"
        
        inputConv.load_python_weight(reader: reader, scope: prfx, number: &number)
        inputConvBatchNorm.load_python_weight(reader: reader, scope: prfx, number: &number)
        
        invertedBlock0.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock1.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock2.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock3.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock4.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock5.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock6.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock7.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock8.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock9.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock10.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock11.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock12.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock13.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock14.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock15.load_python_weight(reader: reader, scope: prfx, number: &number)
        invertedBlock16.load_python_weight(reader: reader, scope: prfx, number: &number)

        outputConv.load_python_weight(reader: reader, scope: prfx, number: &number)
        outputConvBatchNorm.load_python_weight(reader: reader, scope: prfx, number: &number)
        //outputClassifier.load_python_weight(reader: reader, scope: "layer_with_weights-0/", number: &number)
        
        n += 1
        prfx = ""

        outputClassifier.load_python_weight(reader: reader, scope: prfx, number: &n)
        
        //output.load_python_weight(reader: reader, scope: "", number: &number)
        number = number + n - 2
    }
}
