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

public struct InvertedBottleneckBlock: Layer {
    @noDerivative public var addResLayer: Bool
    @noDerivative public var strides: (Int, Int)
    @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))

    public var conv1: Conv2D<Float>
    public var batchNormConv1: BatchNorm<Float>
    public var dConv: DepthwiseConv2D<Float>
    public var batchNormDConv: BatchNorm<Float>
    public var conv2: Conv2D<Float>
    public var batchNormConv2: BatchNorm<Float>

    public init(
        filters: (Int, Int),
        widthMultiplier: Float,
        depthMultiplier: Int = 6,
        strides: (Int, Int) = (1, 1)
    ) {
        self.strides = strides
        self.addResLayer = filters.0 == filters.1 && strides == (1, 1)

        let filterMult = roundFilterPair(filters: filters, widthMultiplier: widthMultiplier)
        let hiddenDimension = filterMult.0 * depthMultiplier
        conv1 = Conv2D<Float>(
            filterShape: (1, 1, filterMult.0, hiddenDimension),
            strides: (1, 1),
            padding: .same)
        dConv = DepthwiseConv2D<Float>(
            filterShape: (3, 3, hiddenDimension, 1),
            strides: strides,
            padding: strides == (1, 1) ? .same : .valid)
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, hiddenDimension, filterMult.1),
            strides: (1, 1),
            padding: .same)
        batchNormConv1 = BatchNorm(featureCount: hiddenDimension)
        batchNormDConv = BatchNorm(featureCount: hiddenDimension)
        batchNormConv2 = BatchNorm(featureCount: filterMult.1)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let pointwise = relu6(batchNormConv1(conv1(input)))
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



extension InvertedBottleneckBlock: ExportableLayer {
    var nameMappings: [String: String] { [
            "conv1": "conv1",
            "batchNormConv1": "conv1BN",
            "dConv": "dConv",
            "batchNormDConv": "dConvBN",
            "conv2": "conv2",
            "batchNormConv2": "conv2BN",
        ] }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        //print("--InvertedBottleneckBlock:save_weight-- \(scope)")
        conv1.save_weight(scope: scope+"/conv1", tensors: &tensors)
        batchNormConv1.save_weight(scope: scope+"/conv1BN", tensors: &tensors)
        dConv.save_weight(scope: scope+"/dConv", tensors: &tensors)
        batchNormDConv.save_weight(scope: scope+"/dConvBN", tensors: &tensors)
        conv2.save_weight(scope: scope+"/conv2", tensors: &tensors)
        batchNormConv2.save_weight(scope: scope+"/conv2BN", tensors: &tensors)
//        for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
//            print(kp)
//            let obj = self[keyPath: kp]
//            print("\(String(describing: type(of: obj)))")
//        }
    }
}

extension InvertedBottleneckBlock: ImportableLayer {
    mutating func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        //print("InvertedBottleneckBlock.load_weight  <<<  \(scope)")
        conv1.load_weight(reader: reader, config: config, scope: scope+"/conv1")
        batchNormConv1.load_weight(reader: reader, config: config, scope: scope+"/conv1BN")
        dConv.load_weight(reader: reader, config: config, scope: scope+"/dConv")
        batchNormDConv.load_weight(reader: reader, config: config, scope: scope+"/dConvBN")
        conv2.load_weight(reader: reader, config: config, scope: scope+"/conv2")
        batchNormConv2.load_weight(reader: reader, config: config, scope: scope+"/conv2BN")
    }
}

extension InvertedBottleneckBlock: SummaryLayer {
    func summary(scope: String) {
        print("InvertedBottleneckBlock:                                               \(scope)")
        print("-------------")
        self.conv1.summary(scope: scope+"/conv1")
        self.batchNormConv1.summary(scope: scope+"/conv1BN")
        if self.strides != (1, 1) {
            self.zeroPad.summary(scope: scope+"/zeroPad")
        }
        self.dConv.summary(scope: scope+"/dConv")
        self.batchNormDConv.summary(scope: scope+"/dConvBN")
        self.conv2.summary(scope: scope+"/conv2")
        self.batchNormConv2.summary(scope: scope+"/conv2BN")
    }
}

extension InvertedBottleneckBlock: ImportablePythonLayer {
    mutating public func load_python_weight(reader: CheckpointReader, scope: String, number: inout Int) {
        //print("--InvertedBottleneckBlock:load_python_weight-- \(scope)")
        conv1.load_python_weight(reader: reader, scope: scope+"/conv1", number: &number)
        batchNormConv1.load_python_weight(reader: reader, scope: scope+"/conv1BN", number: &number)
        dConv.load_python_weight(reader: reader, scope: scope+"/dConv", number: &number)
        batchNormDConv.load_python_weight(reader: reader, scope: scope+"/dConvBN", number: &number)
        conv2.load_python_weight(reader: reader, scope: scope+"/conv2", number: &number)
        batchNormConv2.load_python_weight(reader: reader, scope: scope+"/conv2BN", number: &number)
    }
}
