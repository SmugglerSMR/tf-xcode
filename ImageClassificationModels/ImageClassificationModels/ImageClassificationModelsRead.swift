//
//  ImageClassificationModelsRead.swift
//  ImageClassificationModels
//
//  Created by Рамиль Садыков on 19.06.2020.
//  Copyright © 2020 Sadukow. All rights reserved.
//

import Foundation

import ModelSupport
import TensorFlow

import Python

public struct TransformerLMConfig: Codable {
    public let vocabSize: Int
    public let contextSize: Int
    public let embeddingSize: Int
    public let headCount: Int
    public let layerCount: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "n_vocab"
        case contextSize = "n_ctx"
        case embeddingSize = "n_embd"
        case headCount = "n_head"
        case layerCount = "n_layer"
    }
}

private func checkShapes(_ tensor1: Tensor<Float>, _ tensor2: Tensor<Float>) {
    guard tensor1.shape == tensor2.shape else {
        print("Shape mismatch: \(tensor1.shape) != \(tensor2.shape)")
        fatalError()
    }
}


extension CheckpointReader {
    func readTensor<Scalar: TensorFlowScalar>(
        name: String
    ) -> Tensor<Scalar> {
        return Tensor<Scalar>(loadTensor(named: name))
    }
}

protocol ImportableLayer {
    mutating func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String)
}







extension GlobalAvgPool2D: ImportableLayer {
    var labelMap: String  { return "avgPool" }
    mutating func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        //print("GlobalAvgPool2D.load_weight  <<<  \(scope)")
        //outputClassifier.load_weight(reader: reader, config: config, scope: scope+"/outputClassifier")
    }
}

extension ZeroPadding2D: ImportableLayer {
    var labelMap: String  { return "zeroPad" }
    mutating func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        //print("ZeroPadding2D.load_weight  <<<  \(scope)")
        //self.padding = reader.readTensor(name: scope + "/p")
    }
}

extension BatchNorm {
    func load_weight(reader: CheckpointReader, labels: [String], tensor: inout Tensor<Scalar>)  {
        for label in labels {
            if (reader.containsTensor(named: label)) {
                let new:Tensor<Scalar> = reader.readTensor(name: label)
                guard tensor.shape == new.shape else {
                    print("BatchNorm: shape mismatch: \(tensor.shape) != \(new.shape)")
                    return
                }
                tensor = new
                return
            }
        }
        print("ERROR:  BatchNorm: \(labels) - no find in checkpoint")
        return
    }
    func load_weight_1(reader: CheckpointReader, labels: [String], tensor: inout Tensor<Scalar>)  {
        for label in labels {
            if (reader.containsTensor(named: label)) {
                tensor = reader.readTensor(name: label)
                return
            }
        }
        print("ERROR:  BatchNorm: \(labels) - no find in checkpoint")
        return
    }
}


extension BatchNorm: ImportableLayer {
    mutating func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        
        load_weight( reader: reader,
                     labels: [scope + "/w"],
                     tensor: &scale )

        load_weight( reader: reader,
                     labels: [scope + "/b"],
                     tensor: &offset )

        load_weight_1( reader: reader,
                       labels: [scope + "/m"],
                       tensor: &runningMean.value )

        load_weight_1( reader: reader,
                       labels: [scope + "/v"],
                       tensor: &runningVariance.value )

//        let newRunningMean:Tensor<Scalar> = reader.readTensor(name: scope + "/m")
//        runningMean.value = newRunningMean

//        let newRunningVariance:Tensor<Scalar> = reader.readTensor(name: scope + "/v")
//        runningVariance.value = newRunningVariance
    }
}

extension LayerNorm {
    func load_weight(reader: CheckpointReader, labels: [String], tensor: inout Tensor<Scalar>)  {
        for label in labels {
            if (reader.containsTensor(named: label)) {
                let new:Tensor<Scalar> = reader.readTensor(name: label)
                guard tensor.shape == new.shape else {
                    print("LayerNorm: shape mismatch: \(tensor.shape) != \(new.shape)")
                    return
                }
                tensor = new
                return
            }
        }
        print("ERROR:  LayerNorm: \(labels) - no find in checkpoint")
        return
    }
}


extension LayerNorm: ImportableLayer {
    mutating func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        load_weight( reader: reader,
                     labels: [scope + "/w"],
                     tensor: &scale )

        load_weight( reader: reader,
                     labels: [scope + "/b"],
                     tensor: &offset )
    }
}

extension DepthwiseConv2D {
    func load_weight(reader: CheckpointReader, labels: [String], tensor: inout Tensor<Scalar>)  {
        for label in labels {
            if (reader.containsTensor(named: label)) {
                let new:Tensor<Scalar> = reader.readTensor(name: label)
                guard tensor.shape == new.shape else {
                    print("DepthwiseConv2D: shape mismatch: \(tensor.shape) != \(new.shape)")
                    return
                }
                tensor = new
                return
            }
        }
        print("ERROR:  DepthwiseConv2D: \(labels) - no find in checkpoint")
        return
    }
}

extension DepthwiseConv2D:ImportableLayer {
    mutating func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        
        load_weight( reader: reader,
                     labels: [scope + "/w"],
                     tensor: &filter )

        load_weight( reader: reader,
                     labels: [scope + "/b"],
                     tensor: &bias )
    }
}

extension Conv2D {
    func load_weight(reader: CheckpointReader, labels: [String], tensor: inout Tensor<Scalar>)  {
        for label in labels {
            if (reader.containsTensor(named: label)) {
                let new:Tensor<Scalar> = reader.readTensor(name: label)
                guard tensor.shape == new.shape else {
                    print("Conv2D: \(label) - shape mismatch: \(tensor.shape) != \(new.shape)")
                    return
                }
                //print("Conv2D: \(label) - load successful")
                tensor = new
                return
            }
        }
        print("ERROR:  Conv2D: \(labels) - no find in checkpoint")
        return
    }
}


extension Conv2D: ImportableLayer {
    mutating func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
        
        load_weight( reader: reader,
                     labels: [scope + "/w"],
                     tensor: &filter )

        load_weight( reader: reader,
                     labels: [scope + "/b"],
                     tensor: &bias )

    }
}

extension Dense {
    func load_weight(reader: CheckpointReader, labels: [String], tensor: inout Tensor<Scalar>)  {
        for label in labels {
            if (reader.containsTensor(named: label)) {
                let new:Tensor<Scalar> = reader.readTensor(name: label)
                guard tensor.shape == new.shape else {
                    print("Dense: \(label) - shape mismatch: \(tensor.shape) != \(new.shape)")
                    return
                }
                tensor = new
                return
            }
        }
        print("ERROR:  Dense: \(labels) - no find in checkpoint")
        return
    }
}


extension Dense: ImportableLayer {
    mutating func load_weight(reader: CheckpointReader, config: TransformerLMConfig, scope: String) {
  
        load_weight( reader: reader,
                     labels: [scope + "/w"],
                     tensor: &weight )

        load_weight( reader: reader,
                     labels: [scope + "/b"],
                     tensor: &bias )

    }
}


