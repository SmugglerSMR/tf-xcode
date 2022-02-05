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


public struct Inverted_ResNet_Block: Layer {
    @noDerivative public var addResLayer: Bool
    @noDerivative public var flag: Bool
    @noDerivative public var strides: (Int, Int)
    @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))

    public var conv1: Conv2D<Float>
    public var batchNormConv1: BatchNorm<Float>
    public var dConv: DepthwiseConv2D<Float>
    public var batchNormDConv: BatchNorm<Float>
    public var conv2: Conv2D<Float>
    public var batchNormConv2: BatchNorm<Float>

    public var prefix: String
    
    public init(
        filters: (Int, Int),
        alpha: Float,
        expansion: Int = 6,
        strides: (Int, Int) = (1, 1),
        id: Int = 0
    ) {
        
        
        self.strides = strides
        self.addResLayer = filters.0 == filters.1 && strides == (1, 1)
        
        if id>0 {
            self.prefix = String(format: "block_%d", id)
            self.flag = true
        }
        else {
            self.prefix = "expanded"
            self.flag = false
        }

        let filterMult = roundFilterPair(filters: filters, widthMultiplier: alpha)
        let hiddenDimension = filterMult.0 * expansion
        
        conv1 = Conv2D<Float>( filterShape: (1, 1, filterMult.0, hiddenDimension),
                               strides: (1, 1),
                               padding: .same)
        batchNormConv1 = BatchNorm(featureCount: hiddenDimension,
                                   axis: -1,
                                   momentum: 0.999,
                                   epsilon: 0.001)

        dConv = DepthwiseConv2D<Float>( filterShape: (3, 3, hiddenDimension, 1),
                                        strides: strides,
                                        padding: strides == (1, 1) ? .same : .valid)
        batchNormDConv = BatchNorm(featureCount: hiddenDimension,
                                   axis: -1,
                                   momentum: 0.999,
                                   epsilon: 0.001)

        conv2 = Conv2D<Float>( filterShape: (1, 1, hiddenDimension, filterMult.1),
                               strides: (1, 1),
                               padding: .same)
        batchNormConv2 = BatchNorm(featureCount: filterMult.1,
                                   axis: -1,
                                   momentum: 0.999,
                                   epsilon: 0.001)

    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        
        var pointwise: Tensor<Float>
        if self.flag {
           pointwise = relu6(batchNormConv1(conv1(input)))
        }
        else {
           pointwise = input
        }
        
        var depthwise: Tensor<Float>
        if self.strides == (1, 1) {
            depthwise = relu6(batchNormDConv(dConv(pointwise)))
        } else {
            depthwise = relu6(batchNormDConv(dConv(zeroPad(pointwise))))
        }
        let pointwiseLinear = batchNormConv2(conv2(depthwise))

        if self.addResLayer {
            return input + pointwiseLinear
        } else {
            return pointwiseLinear
        }
    }
}

extension Inverted_ResNet_Block: SummaryLayer {
    func summary(scope: String) {
        print("Inverted_ResNet_Block:                                                 \(scope)\(self.prefix)")
        print("-------------")
        if self.flag {
            self.conv1.summary(scope: scope+self.prefix+"/conv1")
            self.batchNormConv1.summary(scope: scope+self.prefix+"/conv1BN")
            ReLU_summary(scope: scope+self.prefix+"/conv1ReLU")
        }
        if self.strides != (1, 1) {
            self.zeroPad.summary(scope: scope+self.prefix+"/zeroPad")
        }
        self.dConv.summary(scope: scope+self.prefix+"/dConv")
        self.batchNormDConv.summary(scope: scope+self.prefix+"/dConvBN")
        ReLU_summary(scope: scope+self.prefix+"/dConvReLU")
        self.conv2.summary(scope: scope+self.prefix+"/conv2")
        self.batchNormConv2.summary(scope: scope+self.prefix+"/conv2BN")
        if self.addResLayer {
            Add_summary(scope: scope+self.prefix+"/add")
        }
    }
}

extension Inverted_ResNet_Block: ExportableLayer {
    var nameMappings: [String: String] { [
            "conv1": "conv1",
            "batchNormConv1": "conv1BN",
            "dConv": "dConv",
            "batchNormDConv": "dConvBN",
            "conv2": "conv2",
            "batchNormConv2": "conv2BN",
        ] }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        //print("--Inverted_ResNet_Block:save_weight-- \(scope)\(self.prefix)")
        if self.flag {
            conv1.save_weight(scope: scope+self.prefix+"/conv1", tensors: &tensors)
            batchNormConv1.save_weight(scope: scope+self.prefix+"/conv1BN", tensors: &tensors)
        }
        dConv.save_weight(scope: scope+self.prefix+"/dConv", tensors: &tensors)
        batchNormDConv.save_weight(scope: scope+self.prefix+"/dConvBN", tensors: &tensors)
        conv2.save_weight(scope: scope+self.prefix+"/conv2", tensors: &tensors)
        batchNormConv2.save_weight(scope: scope+self.prefix+"/conv2BN", tensors: &tensors)
    }
}

extension Inverted_ResNet_Block: ImportableLayer {
    mutating func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        //print("Inverted_ResNet_Block:load_weight  <<<  \(scope)\(self.prefix)")
        if self.flag {
            conv1.load_weight(reader: reader, config: config, scope: scope+self.prefix+"/conv1")
            batchNormConv1.load_weight(reader: reader, config: config, scope: scope+self.prefix+"/conv1BN")
        }
        dConv.load_weight(reader: reader, config: config, scope: scope+self.prefix+"/dConv")
        batchNormDConv.load_weight(reader: reader, config: config, scope: scope+self.prefix+"/dConvBN")
        conv2.load_weight(reader: reader, config: config, scope: scope+self.prefix+"/conv2")
        batchNormConv2.load_weight(reader: reader, config: config, scope: scope+self.prefix+"/conv2BN")
    }
}

extension Inverted_ResNet_Block: ImportablePythonLayer {
    mutating public func load_python_weight(reader: CheckpointReader, scope: String, number: inout Int) {
        //print("--Inverted_ResNet_Block:load_python_weight-- \(scope)")
        if self.flag {
            conv1.load_python_weight(reader: reader, scope: scope, number: &number)
            batchNormConv1.load_python_weight(reader: reader, scope: scope, number: &number)
        }
        dConv.load_python_weight(reader: reader, scope: scope, number: &number)
        batchNormDConv.load_python_weight(reader: reader, scope: scope, number: &number)
        conv2.load_python_weight(reader: reader, scope: scope, number: &number)
        batchNormConv2.load_python_weight(reader: reader, scope: scope, number: &number)
    }
}
