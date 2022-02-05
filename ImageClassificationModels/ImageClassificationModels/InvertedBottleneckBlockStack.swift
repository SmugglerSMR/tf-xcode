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



public struct InvertedBottleneckBlockStack: Layer {
    var blocks: [InvertedBottleneckBlock] = []

    public init(
        filters: (Int, Int),
        widthMultiplier: Float,
        blockCount: Int,
        initialStrides: (Int, Int) = (2, 2)
    ) {
        self.blocks = [
            InvertedBottleneckBlock(
                filters: (filters.0, filters.1), widthMultiplier: widthMultiplier,
                strides: initialStrides)
        ]
        for _ in 1..<blockCount {
            self.blocks.append(
                InvertedBottleneckBlock(
                    filters: (filters.1, filters.1), widthMultiplier: widthMultiplier)
            )
        }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return blocks.differentiableReduce(input) { $1($0) }
    }
}

extension InvertedBottleneckBlockStack: ExportableLayer {
    var nameMappings: [String: String] { ["blocks": "blocks"] }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        //print("--InvertedBottleneckBlockStack:save_weight-- \(scope)")
        for (index, block) in blocks.enumerated() {
            blocks[index].save_weight(scope: scope + "/blocks/h\(index)", tensors: &tensors)
        }
    }
}

extension InvertedBottleneckBlockStack: ImportableLayer {
    //var nameMappings: [String: String] { ["blocks": "blocks"] }
    mutating func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        //print("\nInvertedBottleneckBlockStack.load_weight  <<<  \(scope)  blocks: \(blocks.count)")
        for (index, block) in blocks.enumerated() {
            blocks[index].load_weight(reader: reader, config: config, scope: scope + "/blocks/h\(index)")
        }
    }
}

extension InvertedBottleneckBlockStack: SummaryLayer {
    func summary(scope: String) {
        print("InvertedBottleneckBlockStack:   \(blocks.count)                                      \(scope)")
        print("-------------")
        for (index, block) in blocks.enumerated() {
            blocks[index].summary(scope: scope + "/blocks/h\(index)")
        }
    }
}

extension InvertedBottleneckBlockStack: ImportablePythonLayer {
    mutating public func load_python_weight(reader: CheckpointReader, scope: String, number: inout Int) {
        //print("--InvertedBottleneckBlockStack:load_python_weight-- \(scope)")
        for (index, block) in blocks.enumerated() {
            blocks[index].load_python_weight(reader: reader, scope: scope + "/blocks/h\(index)", number: &number)
        }
    }
}
