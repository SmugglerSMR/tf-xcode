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

public struct InitialInvertedBottleneckBlock: Layer {
    public var dConv: DepthwiseConv2D<Float>
    public var batchNormDConv: BatchNorm<Float>
    public var conv2: Conv2D<Float>
    public var batchNormConv: BatchNorm<Float>

    public init(filters: (Int, Int), widthMultiplier: Float) {
        let filterMult = roundFilterPair(filters: filters, widthMultiplier: widthMultiplier)
        dConv = DepthwiseConv2D<Float>(
            filterShape: (3, 3, filterMult.0, 1),
            strides: (1, 1),
            padding: .same)
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, filterMult.0, filterMult.1),
            strides: (1, 1),
            padding: .same)
        batchNormDConv = BatchNorm(featureCount: filterMult.0)
        batchNormConv = BatchNorm(featureCount: filterMult.1)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let depthwise = relu6(batchNormDConv(dConv(input)))
        return batchNormConv(conv2(depthwise))
    }
}

extension InitialInvertedBottleneckBlock: ExportableLayer {
    var nameMappings: [String: String] { [
                        "dConv": "dConv",
                        "batchNormDConv": "dConvBN",
                        "conv2": "conv2",
                        "batchNormConv": "convBN",
                      ] }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        //print("--InitialInvertedBottleneckBlock:save_weight-- \(scope)")
        dConv.save_weight(scope: scope+"/dConv", tensors: &tensors)
        batchNormDConv.save_weight(scope: scope+"/dConvBN", tensors: &tensors)
        conv2.save_weight(scope: scope+"/conv2", tensors: &tensors)
        batchNormConv.save_weight(scope: scope+"/convBN", tensors: &tensors)
    }
}

extension InitialInvertedBottleneckBlock: ImportableLayer {
    mutating func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        //print("InitialInvertedBottleneckBlock.load_weight  <<<  \(scope)")
        dConv.load_weight(reader: reader, config: config, scope: scope+"/dConv")
        batchNormDConv.load_weight(reader: reader, config: config, scope: scope+"/dConvBN")
        conv2.load_weight(reader: reader, config: config, scope: scope+"/conv2")
        batchNormConv.load_weight(reader: reader, config: config, scope: scope+"/convBN")
    }
}

extension InitialInvertedBottleneckBlock: SummaryLayer{
    func summary(scope: String) {
        print("InitialInvertedBottleneckBlock:                                        \(scope)")
        print("-------------")
        
        self.dConv.summary(scope: scope+"/dConv")
        self.batchNormDConv.summary(scope: scope+"/dConvBN")
        self.conv2.summary(scope: scope+"/conv2")
        self.batchNormConv.summary(scope: scope+"/convBN")

    }
}

extension InitialInvertedBottleneckBlock: ImportablePythonLayer {
    mutating public func load_python_weight(reader: CheckpointReader, scope: String, number: inout Int) {
        //print("--InitialInvertedBottleneckBlock:load_python_weight-- \(scope)")
        dConv.load_python_weight(reader: reader, scope: scope+"/dConv", number: &number)
        batchNormDConv.load_python_weight(reader: reader, scope: scope+"/dConvBN", number: &number)
        conv2.load_python_weight(reader: reader, scope: scope+"/conv2", number: &number)
        batchNormConv.load_python_weight(reader: reader, scope: scope+"/convBN", number: &number)
    }
}
